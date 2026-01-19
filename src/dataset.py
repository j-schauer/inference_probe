"""
Dataset loader for probe-based slot extraction.

Converts compiled examples (character spans) to tokenized training examples
with query positions, valid masks, and token-level labels.
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
    Convert character span to token span using offset mapping.

    Args:
        offsets: List of (char_start, char_end) for each token
        char_start: Character start position (inclusive)
        char_end: Character end position (exclusive)

    Returns:
        (token_start, token_end) both inclusive, or None if no overlap
    """
    tok_indices = []
    for i, (s, e) in enumerate(offsets):
        if s == e:  # Skip empty spans (special tokens)
            continue
        # Check overlap: NOT (token ends before span OR token starts after span)
        if not (e <= char_start or s >= char_end):
            tok_indices.append(i)
    return (min(tok_indices), max(tok_indices)) if tok_indices else None


class ProbeDataset(Dataset):
    """
    Dataset for probe-based slot extraction training.

    Each example contains:
    - input_ids: Full tokenized sequence
    - attention_mask: 1s for real tokens
    - valid_mask: True only for utterance tokens (probe can only point here)
    - query_positions: Token index for each slot marker
    - labels: Presence flag + start/end indices per slot
    """

    def __init__(
        self,
        compiled_file: str,
        schema_file: str,
        tokenizer
    ):
        """
        Args:
            compiled_file: Path to compiled_clean_shuffled.jsonl
            schema_file: Path to function_schemas.json
            tokenizer: HuggingFace tokenizer
        """
        self.tokenizer = tokenizer

        with open(schema_file) as f:
            self.schemas = json.load(f)

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
        ex = self.examples[idx]
        return self.build_training_example(ex)

    def build_training_example(self, ex: Dict) -> Dict:
        """
        Build a single training example from compiled data.

        Two-pass tokenization:
        1. Tokenize utterance alone to get char->token mapping
        2. Build full sequence and track positions
        """
        func_name = ex["function"]
        utterance = ex["utterance"]
        present_slots = ex["slots"]

        schema_slots = sorted(self.schemas[func_name]["slots"].keys())

        # === PASS 1: Tokenize utterance for char->token mapping ===
        utt_enc = self.tokenizer(
            utterance,
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        utt_ids = utt_enc["input_ids"]
        utt_offsets = utt_enc["offset_mapping"]

        # Convert char spans to token spans
        token_spans = {}
        for slot_name, slot_data in present_slots.items():
            span = char_span_to_token_span(
                utt_offsets,
                slot_data["char_start"],
                slot_data["char_end"]
            )
            if span:
                token_spans[slot_name] = span

        # === PASS 2: Build full sequence ===

        # Function prefix
        func_prefix = f"Function: {func_name}\n"
        func_ids = self.tokenizer(func_prefix, add_special_tokens=False)["input_ids"]

        # Separator
        sep_ids = self.tokenizer("\n\n", add_special_tokens=False)["input_ids"]

        # Build input_ids
        input_ids = []
        input_ids.extend(func_ids)

        utterance_start = len(input_ids)
        input_ids.extend(utt_ids)
        utterance_end = len(input_ids)

        input_ids.extend(sep_ids)

        # Add slot markers and track query positions
        query_positions = {}
        for slot_name in schema_slots:
            marker = f"{slot_name}:"
            marker_ids = self.tokenizer(marker, add_special_tokens=False)["input_ids"]
            input_ids.extend(marker_ids)
            query_positions[slot_name] = len(input_ids) - 1  # Last token of marker

            newline_ids = self.tokenizer("\n", add_special_tokens=False)["input_ids"]
            input_ids.extend(newline_ids)

        # === Build masks and labels ===

        seq_len = len(input_ids)
        attention_mask = [1] * seq_len

        # Valid mask: True only for utterance tokens
        valid_mask = [False] * seq_len
        for i in range(utterance_start, utterance_end):
            valid_mask[i] = True

        # Labels for each schema slot
        labels = {}
        for slot_name in schema_slots:
            if slot_name in token_spans:
                start, end = token_spans[slot_name]
                labels[slot_name] = {
                    "present": True,
                    "start": start,  # Utterance-relative
                    "end": end       # Utterance-relative
                }
            else:
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
    Create a collate function for DataLoader.

    Args:
        pad_token_id: Token ID to use for padding

    Returns:
        Collate function that batches examples with padding
    """
    def collate_fn(batch: List[Dict]) -> Dict:
        max_len = max(len(ex["input_ids"]) for ex in batch)

        input_ids = []
        attention_mask = []
        valid_mask = []

        for ex in batch:
            seq_len = len(ex["input_ids"])
            pad_len = max_len - seq_len

            input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(ex["attention_mask"] + [0] * pad_len)
            valid_mask.append(ex["valid_mask"] + [False] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),
            "utterance_start": [ex["utterance_start"] for ex in batch],
            "utterance_end": [ex["utterance_end"] for ex in batch],
            "query_positions": [ex["query_positions"] for ex in batch],
            "labels": [ex["labels"] for ex in batch],
            "function": [ex["function"] for ex in batch],
        }

    return collate_fn
