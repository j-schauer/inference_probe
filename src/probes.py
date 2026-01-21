"""
Probe head definitions for slot extraction.

WHAT ARE PROBES?
    Probes are simple classifiers that read information from a neural network's
    internal representations. The idea: if a simple classifier can extract
    information X from hidden states, then the network has learned to encode X.

    We use two probes:
    - NullHead: "Is this slot mentioned?" (yes/no) - a linear classifier
    - PointerHead: "Where is the value?" (which tokens) - bilinear attention

WHY SIMPLE?
    If we used a complex probe (like a deep neural network), it might be
    computing new information rather than just reading what's already there.
    Linear and bilinear probes are simple enough that they can only READ,
    not COMPUTE. This means if they work, the LLM truly encoded the answer.
"""

import math
import torch
import torch.nn as nn


class NullHead(nn.Module):
    """
    A LINEAR PROBE for slot presence detection.

    This is the simplest possible classifier: a single matrix multiplication.

        query [4096] → Linear → [2] logits → argmax → present/absent

    The weight matrix W has shape [2, 4096]. Each row is a "detector":
    - Row 0 detects "present-ness"
    - Row 1 detects "absent-ness"

    The dot product of query with each row gives a score. Whichever is
    higher wins. That's it - no hidden layers, no nonlinearities.

    ~8,000 parameters total (2 × 4096 + 2 bias).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # Linear(in_features, out_features) = weight [out, in] + bias [out]
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, query_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_vec: [4096] or [batch, 4096] - hidden state at slot marker

        Returns:
            logits: [2] or [batch, 2] - scores for [present, absent]
            Use argmax to get prediction: 0=present, 1=absent
        """
        if query_vec.dim() == 1:
            query_vec = query_vec.unsqueeze(0)
        return self.classifier(query_vec)


class PointerHead(nn.Module):
    """
    A BILINEAR PROBE for finding where slot values are in the input.

    WHY BILINEAR?
        We need to compare TWO vectors:
        - The "query" (hidden state at "origin:") - encodes "what am I looking for?"
        - Each position's hidden state - encodes "what is here?"

        A linear probe can only transform ONE vector. Bilinear lets us compare two.

    HOW IT WORKS:
        1. Transform the query:  transformed = W @ query        [4096] → [4096]
        2. Compare to each position:  score[i] = transformed · hidden[i]   → scalar
        3. Do this for all positions:  scores = [score_0, score_1, ..., score_T]
        4. Pick the winner:  argmax(scores) → position index

        We have TWO weight matrices (W_start and W_end) because the "start of a value"
        and "end of a value" are different concepts. They're trained independently.

    THE MATH:
        score = query @ W @ hidden[position]
                [1,d]  [d,d]    [d,1]
                    ↓
                  scalar

        W is [4096, 4096] - that's ~16M parameters per matrix, ~33M total.

    WHY NOT Linear(4096, max_seq_len)?
        That would learn "position 5 is often the answer" - but position 5 means
        different things in different inputs! We need to compare against the CONTENT
        at each position, not just learn position-specific patterns.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # W_start and W_end are [hidden_size, hidden_size] = [4096, 4096]
        # bias=False because we only need the transformation, not an offset
        self.W_start = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_end = nn.Linear(hidden_size, hidden_size, bias=False)
        # Scale factor to prevent dot products from getting too large
        self.scale = 1.0 / math.sqrt(hidden_size)

    def forward(
        self,
        query_vec: torch.Tensor,
        all_hidden: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_vec: [batch, 4096] - hidden state at slot marker (e.g., "origin:")
            all_hidden: [batch, seq_len, 4096] - hidden states at ALL positions
            valid_mask: [batch, seq_len] - True for positions we can point to (utterance only)

        Returns:
            start_logits: [batch, seq_len] - score for each position being the START
            end_logits: [batch, seq_len] - score for each position being the END
            Use argmax to get predictions.
        """
        if query_vec.dim() == 1:
            query_vec = query_vec.unsqueeze(0)
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()

        # THE BILINEAR COMPUTATION:
        #
        # Step 1: Transform query with learned W matrix
        #   W_start(query_vec) → [batch, 4096]
        #
        # Step 2: Dot product with each position's hidden state
        #   einsum "bh,bth->bt" means:
        #     for each batch b:
        #       for each position t:
        #         score[b,t] = sum over h of (transformed_query[b,h] * hidden[b,t,h])
        #   Result: [batch, seq_len] scores
        #
        start_logits = torch.einsum("bh,bth->bt", self.W_start(query_vec), all_hidden) * self.scale
        end_logits = torch.einsum("bh,bth->bt", self.W_end(query_vec), all_hidden) * self.scale

        # Mask out invalid positions (function prefix, slot markers, padding)
        # -1e4 is effectively negative infinity for softmax/argmax
        start_logits = start_logits.masked_fill(~valid_mask, -1e4)
        end_logits = end_logits.masked_fill(~valid_mask, -1e4)

        return start_logits, end_logits


def load_probe_heads(path: str, device: torch.device = None) -> tuple[NullHead, PointerHead]:
    """
    Load trained probe heads from checkpoint.

    Args:
        path: Path to probe_heads.pt file
        device: Device to load to (default: cuda if available)

    Returns:
        null_head, pointer_head: Loaded and initialized probe heads
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(path, map_location=device)
    hidden_size = state["null_head"]["classifier.weight"].shape[1]

    null_head = NullHead(hidden_size).to(device)
    pointer_head = PointerHead(hidden_size).to(device)

    null_head.load_state_dict(state["null_head"])
    pointer_head.load_state_dict(state["pointer_head"])

    null_head.eval()
    pointer_head.eval()

    return null_head, pointer_head
