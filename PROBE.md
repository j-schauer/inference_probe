# Probe Implementation Details

This document describes the specific implementation of probe-based slot extraction using Llama 3.1 8B and HuggingFace. For general background on probing, see [SCIENCE.md](SCIENCE.md).

## Goal

Extract slot values from natural language by pointing to token positions, not generating text.

```
Input:  "Book a flight from Seattle to Denver tomorrow"

Output: origin → tokens[4:4] → "Seattle"
        destination → tokens[6:6] → "Denver"
        date → tokens[7:7] → "tomorrow"
```

One forward pass through the model, ~20ms inference.

## Architecture Overview

```
┌─────────────────────────────────────────┐
│      Llama 3.1 8B (frozen weights)      │
│      + LoRA adapters (trainable)        │
└─────────────────┬───────────────────────┘
                  │
          Hidden States [T, 4096]
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌────────┐   ┌────────┐   ┌────────┐
│NullHead│   │NullHead│   │NullHead│   (one query per slot)
└───┬────┘   └───┬────┘   └───┬────┘
    │            │            │
┌───┴─────┐ ┌───┴─────┐ ┌───┴─────┐
│PointerHd│ │PointerHd│ │PointerHd│
└───┬─────┘ └───┴─────┘ └───┬─────┘
    │            │            │
    ▼            ▼            ▼
origin:4-4   dest:absent   date:7-7
```

## Probe Heads

### NullHead: Presence Classifier

Determines if a slot value exists in the utterance.

```python
class NullHead(nn.Module):
    def __init__(self, hidden_size):  # 4096 for Llama 8B
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, query_vec):
        return self.classifier(query_vec)  # [present_logit, absent_logit]
```

- **Parameters**: 4096 × 2 + 2 = 8,194
- **Output**: argmax=0 means present, argmax=1 means absent

### PointerHead: Span Extractor

Points to start and end token positions.

```python
class PointerHead(nn.Module):
    def __init__(self, hidden_size):
        self.W_start = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_end = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = 1.0 / math.sqrt(hidden_size)  # 1/64

    def forward(self, query_vec, all_hidden, valid_mask):
        # Bilinear attention: query compatibility with each position
        start_logits = einsum("bh,bth->bt", self.W_start(query_vec), all_hidden) * self.scale
        end_logits = einsum("bh,bth->bt", self.W_end(query_vec), all_hidden) * self.scale

        # Mask non-utterance positions (function prefix, slot markers)
        start_logits = start_logits.masked_fill(~valid_mask, -1e4)
        end_logits = end_logits.masked_fill(~valid_mask, -1e4)

        return start_logits, end_logits
```

- **Parameters**: 2 × (4096 × 4096) = 33,554,432
- **Output**: argmax over positions gives start/end indices

### Why Bilinear?

The PointerHead uses bilinear attention: `score = query @ W @ position`. This learns a compatibility function between "what we're looking for" (query) and "what's at each position" (hidden states).

The scale factor (`1/sqrt(4096)`) prevents dot products from exploding - same technique used in transformer attention.

### Why No Bias in PointerHead?

PointerHead has `bias=False`. This makes it scale-invariant - if hidden states are scaled by factor k, all positions scale equally, preserving the argmax. This proved critical for deployment (see appendix/VLLM_DEPLOYMENT.md).

NullHead has bias, which caused issues when hidden state scales changed between training and inference.

## Input Format

```
Function: book_restaurant
Book a table for 4 at 7pm

party_size:
restaurant_name:
time:
```

Components:
1. **Function prefix**: `Function: {name}\n` - identifies which function's slots to extract
2. **Utterance**: The user's natural language input
3. **Separator**: `\n\n`
4. **Slot markers**: One per schema slot, alphabetically sorted, format `{slot_name}:\n`

The **query position** for each slot is the last token of its marker (the colon). The probe reads the hidden state at this position to determine the slot's value.

All schema slots are included, not just those present in the utterance. The NullHead learns which slots are absent.

## Data

### Source

Salesforce xLAM Function Calling 60K dataset, filtered and processed:

1. Filter to top 50 functions (23,106 examples)
2. Remove 4 problematic functions with schema inconsistencies
3. Remove multi-call examples (keep single function calls only)
4. Convert to probe format with character spans

**Final dataset**: 9,860 examples, 45 functions, 99 total slots

### Format

```json
{
  "function": "book_restaurant",
  "utterance": "Book a table for 4 at 7pm",
  "slots": {
    "party_size": {"char_start": 18, "char_end": 19, "value": "4"},
    "time": {"char_start": 23, "char_end": 26, "value": "7pm"}
  }
}
```

Character spans are converted to token spans at training time using the tokenizer's offset mapping.

### Split

- **Train**: 8,874 examples (90%)
- **Holdout**: 986 examples (10%)
- Pre-shuffled with seed 42 for reproducibility

See `data/DATA_CARD.md` for full provenance.

## Training

### What Gets Trained

| Component | Parameters | Trainable |
|-----------|------------|-----------|
| Llama 8B backbone | ~8B | Frozen |
| LoRA adapters | ~8M | Yes |
| NullHead | ~8K | Yes |
| PointerHead | ~33M | Yes |

Total trainable: ~41M parameters (0.5% of model)

### LoRA Configuration

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
```

LoRA adapts attention layers so the model learns to encode slot information at query positions. Without LoRA, probes still work (~90%) but less accurately (~98% with LoRA).

### Loss Function

```python
total_loss = 0

for slot in schema_slots:
    query = hidden[query_position[slot]]

    # NullHead loss (always computed)
    null_logits = null_head(query)
    target = 0 if slot_present else 1
    total_loss += cross_entropy(null_logits, target)

    # PointerHead loss (only if slot is present)
    if slot_present:
        start_logits, end_logits = pointer_head(query, hidden, valid_mask)
        total_loss += cross_entropy(start_logits, gold_start)
        total_loss += cross_entropy(end_logits, gold_end)
```

Cross-entropy treats positions as classes. The model learns to increase the logit at the correct position while decreasing others.

### Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        # Forward through LLM + LoRA
        hidden = model(input_ids).last_hidden_state

        # Compute loss over all slots in batch
        loss = compute_loss(hidden, batch)

        loss.backward()
        clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
```

- **Optimizer**: AdamW, lr=2e-5
- **Batch size**: 4 (limited by GPU memory)
- **Epochs**: 1 (sufficient for convergence)
- **Gradient clipping**: max_norm=1.0

### Hidden State Extraction

```python
def get_hidden_states(model, input_ids, attention_mask):
    # Unwrap PEFT model to get LlamaModel
    base = model.base_model.model
    llama = base.model

    outputs = llama(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        return_dict=True,
    )

    return outputs.last_hidden_state  # Post-RMSNorm, [B, T, 4096]
```

We explicitly use `last_hidden_state` rather than `hidden_states[-1]` to ensure we get post-final-RMSNorm hidden states consistently.

## Code Files

### `src/dataset.py`

Handles data loading and tokenization.

**Key functions**:
- `ProbeDataset`: Loads JSONL, builds training examples
- `char_span_to_token_span()`: Converts character offsets to token indices
- `make_collate_fn()`: Batches examples with proper padding

**Input**: `compiled_clean_shuffled.jsonl` + `function_schemas.json`

**Output**: Training examples with:
- `input_ids`: Full tokenized sequence
- `valid_mask`: True only for utterance tokens
- `query_positions`: Token index for each slot marker
- `labels`: Presence flag + start/end indices per slot

### `src/probes.py`

Defines the probe head classes.

**Classes**:
- `NullHead`: Linear(4096, 2) classifier
- `PointerHead`: Bilinear attention with W_start, W_end

These are standalone PyTorch modules, independent of the LLM architecture.

### `src/train.py`

Main training script.

**Flow**:
1. Load Llama 8B + apply LoRA
2. Initialize probe heads (random weights)
3. Load dataset
4. Training loop (forward → loss → backward → step)
5. Evaluate on holdout
6. Save LoRA adapter + probe heads

**Outputs**:
- `lora_adapter/`: PEFT format LoRA weights
- `probe_heads.pt`: NullHead + PointerHead state dicts

## Results

### Llama 3.1 8B Instruct

| Metric | Value |
|--------|-------|
| Training examples | 8,874 |
| Holdout examples | 986 |
| Epochs | 1 |
| Presence accuracy | 97.4% |
| **Exact match** | **98.6%** |
| Training time | ~15 minutes (A100) |

### Error Analysis

Most errors fall into:
- **Boundary errors**: Off-by-one on multi-token values
- **Ambiguous spans**: Multiple valid extractions ("the 4th" vs "4th")
- **Tokenization artifacts**: Subword boundaries don't align with value boundaries

The 98% accuracy is likely near the ceiling for this task given annotation ambiguity.

## Beyond HuggingFace

This project also explored:
- **vLLM deployment**: Required fixes for hidden state normalization and LoRA loading
- **AWQ quantization**: 4-bit inference with <1% accuracy drop
- **Other models**: Tested Qwen, OLMo, Nemotron

These topics are covered in the appendix but are not necessary for understanding or reproducing the core approach. The HuggingFace training described here is the foundation.

See:
- `appendix/VLLM_DEPLOYMENT.md` - Deployment challenges and solutions
- `appendix/QUANTIZATION.md` - AWQ results
- `appendix/MULTI_MODEL.md` - Cross-model experiments

## Reproducing Results

1. **Get the code**: Clone this repo
2. **Get a GPU**: A100 recommended, A10G works (see `modal/MODAL_GUIDE.md`)
3. **Download base model**: `meta-llama/Llama-3.1-8B-Instruct`
4. **Run training**: `python src/train.py --data data/ --epochs 1`
5. **Check results**: Script prints holdout metrics at the end

Training takes ~15 minutes on A100, ~25 minutes on A10G.
