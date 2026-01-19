"""
Probe head definitions for slot extraction.

Two heads work together:
- NullHead: Binary classifier for slot presence
- PointerHead: Bilinear attention for span extraction
"""

import math
import torch
import torch.nn as nn


class NullHead(nn.Module):
    """
    Binary classifier for slot presence detection.

    Input: Query vector from hidden state at slot marker position
    Output: 2 logits [present, absent]

    Prediction: argmax=0 means present, argmax=1 means absent
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, query_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_vec: [H] or [B, H] hidden state at slot marker position

        Returns:
            logits: [2] or [B, 2] presence/absence logits
        """
        if query_vec.dim() == 1:
            query_vec = query_vec.unsqueeze(0)
        return self.classifier(query_vec)


class PointerHead(nn.Module):
    """
    Bilinear attention head for span extraction.

    Computes compatibility scores between a query vector and all positions
    in the sequence, separately for start and end boundaries.

    Input: Query vector + all hidden states + validity mask
    Output: Start logits and end logits over sequence positions
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_start = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_end = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = 1.0 / math.sqrt(hidden_size)

    def forward(
        self,
        query_vec: torch.Tensor,
        all_hidden: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_vec: [B, H] hidden state at slot marker position
            all_hidden: [B, T, H] all token hidden states
            valid_mask: [B, T] True for positions that can be pointed to

        Returns:
            start_logits: [B, T] scores for start position
            end_logits: [B, T] scores for end position
        """
        if query_vec.dim() == 1:
            query_vec = query_vec.unsqueeze(0)
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()

        # Bilinear attention: query @ W @ hidden_states.T
        # einsum "bh,bth->bt" = batched dot product over H dimension
        start_logits = torch.einsum("bh,bth->bt", self.W_start(query_vec), all_hidden) * self.scale
        end_logits = torch.einsum("bh,bth->bt", self.W_end(query_vec), all_hidden) * self.scale

        # Mask invalid positions (non-utterance tokens)
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
