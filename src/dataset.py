"""
Dataset loader for probe-based slot extraction.

WHAT THIS DOES:
    Converts raw training data into the format needed for probe training.

    Raw data looks like:
        {
            "function": "book_restaurant",
            "utterance": "Book a table for 4 at 7pm",
            "slots": {
                "party_size": {"char_start": 17, "char_end": 18, "value": "4"},
                "time": {"char_start": 22, "char_end": 26, "value": "7pm"}
            }
        }

    We convert this to:
        - input_ids: Token IDs for "Function: book_restaurant\nBook a table for 4...\n\nparty_size:\ntime:\n"
        - valid_mask: True only for utterance tokens (the probe can only point here)
        - query_positions: {"party_size": 47, "time": 52} (token indices of slot markers)
        - labels: {"party_size": {"present": True, "start": 5, "end": 5}, ...}

KEY CHALLENGE: CHARACTER → TOKEN CONVERSION
    The raw data has character positions ("chars 17-18"), but we need token positions
    ("token 5"). Tokenization is not 1-to-1 with characters:
        "7pm" might become ["7", "pm"] (2 tokens) or ["7pm"] (1 token)

    We use the tokenizer's offset_mapping to find which tokens overlap with the
    character span.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def char_span_to_token_span(
    offsets: List[Tuple[int, int]],
    char_start: int,
    char_end: int
) -> Optional[Tuple[int, int]]:
    """
    Convert character positions to token positions.

    The tokenizer tells us which characters each token covers via offset_mapping:
        offsets = [(0,4), (4,5), (5,10), ...]  # token 0 covers chars 0-4, etc.

    We find all tokens that overlap with our character span.

    Args:
        offsets: [(char_start, char_end), ...] for each token
        char_start: Start character (inclusive)
        char_end: End character (exclusive)

    Returns:
        (first_token, last_token) both inclusive, or None if no overlap
    """
    tok_indices = []
    for i, (s, e) in enumerate(offsets):
        if s == e:  # Skip empty spans (special tokens like [CLS])
            continue
        # Check overlap: does this token intersect our character span?
        # NOT (token ends before span starts OR token starts after span ends)
        if not (e <= char_start or s >= char_end):
            tok_indices.append(i)
    return (min(tok_indices), max(tok_indices)) if tok_indices else None


class ProbeDataset(Dataset):
    """
    PyTorch Dataset for probe training.

    Loads raw examples and converts them to training format on-demand.
    Each example becomes a dict with everything needed for compute_loss().
    """

    def __init__(
        self,
        compiled_file: str,
        schema_file: str,
        tokenizer
    ):
        """
        Args:
            compiled_file: JSONL file with examples (one JSON object per line)
            schema_file: JSON file mapping function names → slot definitions
            tokenizer: HuggingFace tokenizer (converts text ↔ token IDs)
        """
        self.tokenizer = tokenizer

        # Load function schemas: {"book_restaurant": {"slots": {"party_size": {...}, ...}}}
        with open(schema_file) as f:
            self.schemas = json.load(f)

        # Load examples, keeping only those with known functions
        self.examples = []
        with open(compiled_file) as f:
            for line in f:
                ex = json.loads(line)
                if ex["function"] in self.schemas:
                    self.examples.append(ex)

        print(f"Loaded {len(self.examples)} examples from {compiled_file}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        """Convert raw example to training format (called by DataLoader)."""
        ex = self.examples[idx]
        return self.build_training_example(ex)

    def build_training_example(self, ex: Dict) -> Dict:
        """
        Convert a raw example into training format.

        INPUT (raw example):
            {
                "function": "book_restaurant",
                "utterance": "Book for 4 at 7pm",
                "slots": {"party_size": {"char_start": 9, "char_end": 10, "value": "4"}}
            }

        OUTPUT (training format):
            {
                "input_ids": [token IDs for full sequence],
                "valid_mask": [False, False, ..., True, True, ..., False, False],
                "query_positions": {"party_size": 47, "time": 52},
                "labels": {"party_size": {"present": True, "start": 2, "end": 2}, ...}
            }

        THE SEQUENCE WE BUILD:
            "Function: book_restaurant\n"     ← function prefix (can't point here)
            "Book for 4 at 7pm"               ← utterance (CAN point here)
            "\n\n"                            ← separator
            "party_size:\n"                   ← slot marker (read query from here)
            "time:\n"                         ← slot marker
        """
        func_name = ex["function"]
        utterance = ex["utterance"]
        present_slots = ex["slots"]  # Slots that have values in this example

        # Get ALL slots for this function (not just the ones present)
        schema_slots = sorted(self.schemas[func_name]["slots"].keys())

        # ================================================================
        # PASS 1: Tokenize utterance alone to get character→token mapping
        # ================================================================
        # We need offset_mapping to convert "chars 9-10" → "token 2"
        utt_enc = self.tokenizer(
            utterance,
            return_offsets_mapping=True,  # Returns [(char_start, char_end), ...] per token
            add_special_tokens=False      # Don't add [CLS], [SEP], etc.
        )
        utt_ids = utt_enc["input_ids"]       # Token IDs for utterance
        utt_offsets = utt_enc["offset_mapping"]  # Character spans per token

        # Convert character spans to token spans
        token_spans = {}
        for slot_name, slot_data in present_slots.items():
            span = char_span_to_token_span(
                utt_offsets,
                slot_data["char_start"],
                slot_data["char_end"]
            )
            if span:
                token_spans[slot_name] = span  # e.g., {"party_size": (2, 2)}

        # ================================================================
        # PASS 2: Build the full input sequence
        # ================================================================
        # We construct: "Function: {name}\n{utterance}\n\n{slot1}:\n{slot2}:\n..."

        # Function prefix (probe can't point here)
        func_prefix = f"Function: {func_name}\n"
        func_ids = self.tokenizer(func_prefix, add_special_tokens=False)["input_ids"]

        # Separator between utterance and slot markers
        sep_ids = self.tokenizer("\n\n", add_special_tokens=False)["input_ids"]

        # Assemble the full sequence
        input_ids = []
        input_ids.extend(func_ids)

        # Track where the utterance starts/ends (for valid_mask)
        utterance_start = len(input_ids)
        input_ids.extend(utt_ids)
        utterance_end = len(input_ids)

        input_ids.extend(sep_ids)

        # Add slot markers and record their positions
        # These are where we read the "query" from during training/inference
        query_positions = {}
        for slot_name in schema_slots:
            marker = f"{slot_name}:"
            marker_ids = self.tokenizer(marker, add_special_tokens=False)["input_ids"]
            input_ids.extend(marker_ids)
            # Query position = last token of marker (the ":")
            query_positions[slot_name] = len(input_ids) - 1

            newline_ids = self.tokenizer("\n", add_special_tokens=False)["input_ids"]
            input_ids.extend(newline_ids)

        # ================================================================
        # Build masks and labels
        # ================================================================

        seq_len = len(input_ids)
        attention_mask = [1] * seq_len  # All real tokens (no padding yet)

        # valid_mask: True ONLY for utterance tokens
        # The probe can only point to positions where valid_mask=True
        # This prevents it from pointing to the function prefix or slot markers
        valid_mask = [False] * seq_len
        for i in range(utterance_start, utterance_end):
            valid_mask[i] = True

        # Labels: for each slot in the schema, record presence and position
        labels = {}
        for slot_name in schema_slots:
            if slot_name in token_spans:
                start, end = token_spans[slot_name]
                labels[slot_name] = {
                    "present": True,
                    "start": start,  # Utterance-relative (add utterance_start for absolute)
                    "end": end
                }
            else:
                # Slot not mentioned in this utterance
                labels[slot_name] = {"present": False}

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "valid_mask": valid_mask,
            "utterance_start": utterance_start,
            "utterance_end": utterance_end,
            "query_positions": query_positions,
            "labels": labels,
            "function": func_name,
        }


def make_collate_fn(pad_token_id: int):
    """
    Create a function that batches examples together for the DataLoader.

    WHY WE NEED THIS:
        Different examples have different sequence lengths. To process them
        as a batch (tensor), we need to pad shorter sequences to match the longest.

        Example batch (before padding):
            Example 1: [101, 2054, 2003, ...]  (47 tokens)
            Example 2: [101, 1045, 2293, ...]  (52 tokens)  ← longest

        After padding (to length 52):
            Example 1: [101, 2054, 2003, ..., PAD, PAD, PAD, PAD, PAD]
            Example 2: [101, 1045, 2293, ...]

        attention_mask tells the model to ignore PAD tokens.
        valid_mask is False for PAD tokens (can't point there).

    Args:
        pad_token_id: Token ID to use for padding (usually tokenizer.pad_token_id)

    Returns:
        A function that takes a list of examples and returns a batched dict
    """
    def collate_fn(batch: List[Dict]) -> Dict:
        # Find the longest sequence in this batch
        max_len = max(len(ex["input_ids"]) for ex in batch)

        input_ids = []
        attention_mask = []
        valid_mask = []

        for ex in batch:
            seq_len = len(ex["input_ids"])
            pad_len = max_len - seq_len

            # Pad sequences to max_len
            input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(ex["attention_mask"] + [0] * pad_len)  # 0 = ignore
            valid_mask.append(ex["valid_mask"] + [False] * pad_len)  # Can't point to padding

        return {
            # Tensors (can be batched)
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),
            # Lists (different structure per example, can't be tensors)
            "utterance_start": [ex["utterance_start"] for ex in batch],
            "utterance_end": [ex["utterance_end"] for ex in batch],
            "query_positions": [ex["query_positions"] for ex in batch],
            "labels": [ex["labels"] for ex in batch],
            "function": [ex["function"] for ex in batch],
        }

    return collate_fn
