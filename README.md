# Probe-Based Slot Extraction

This project extracts slot values from natural language by reading the internal state of a large language model, rather than having it generate text.

```
Input:  "Book a flight from Seattle to Denver tomorrow"

Output: origin=Seattle, destination=Denver, date=tomorrow
```

One forward pass, ~20ms. No text generation, no JSON parsing.

## The Core Idea

When you ask an LLM to extract slots, it already "knows" the answer internally before it starts generating tokens. We tap into that internal knowledge directly using lightweight classifiers called **probes**.

### Why Not Just Generate?

Traditional approach:
```
Input → LLM generates → '{"origin": "Seattle", "destination": "Denver"}' → Parse JSON
```

Problems: Slow (200-500ms), can hallucinate, JSON parsing can fail.

Probe approach:
```
Input → LLM internal state → Probe points to positions → Extract "Seattle", "Denver"
```

Benefits: Fast (20ms), no hallucination (just points to input), no parsing.

## How It Works

### Step 1: Understand What Hidden States Are

When you feed tokens into a transformer, it doesn't just produce output at the end. Every layer produces a vector for every input token position. These vectors are called **hidden states**.

```
Input tokens: ["Book", "a", "flight", "from", "Seattle", ...]
                 ↓      ↓      ↓        ↓        ↓
Layer 1:       [h₀,    h₁,    h₂,      h₃,      h₄,    ...]
                 ↓      ↓      ↓        ↓        ↓
Layer 2:       [h₀,    h₁,    h₂,      h₃,      h₄,    ...]
                 ↓      ↓      ↓        ↓        ↓
  ...
                 ↓      ↓      ↓        ↓        ↓
Layer 32:      [h₀,    h₁,    h₂,      h₃,      h₄,    ...]
```

Each hidden state is a 4096-dimensional vector (for Llama 8B). By layer 32, each vector has "seen" the entire sequence through attention and encodes rich semantic information.

The model processes all positions in parallel (not one at a time like RNNs) - that's what makes transformers fast.

### Step 2: Give the Model a Place to Encode Answers

We construct the input with **slot markers** at the end:

```
Function: book_flight
Book a flight from Seattle to Denver tomorrow

origin:
destination:
date:
```

Why? The model needs specific positions where we can read slot-specific answers. Through attention, the hidden state at `origin:` gathers information about what the origin value is. We read from that position.

### Step 3: Read with Probes

We attach two small neural networks to read from the hidden states:

**NullHead** - "Is this slot present in the utterance?"
- A simple linear classifier: 4096 dimensions → 2 outputs (present/absent)
- Reads from the slot marker position
- Outputs: "yes, origin has a value" or "no, origin is not mentioned"

**PointerHead** - "Where does the value start and end?"
- Compares the slot marker's hidden state against ALL positions
- Outputs: position 4 is the start, position 4 is the end → extract "Seattle"

### Step 4: The Bilinear Trick

The PointerHead uses **bilinear attention** to find where the answer is:

```python
query = hidden_state_at_origin_marker  # [4096] - "what am I looking for?"

# Compare against every position in the sequence
for each position i:
    score[i] = query @ W @ hidden[i]   # How well does position i match?

answer_position = argmax(scores)        # Highest score wins
```

The learned matrix W (4096 × 4096) transforms the query so it can be compared against each position's hidden state. We do this twice - once for start position (W_start) and once for end position (W_end).

Why not just `Linear(4096, max_positions)`? Because position 5 means different things in different inputs. We need to compare against the actual content at each position, not just learn "position 5 is usually the answer."

## Architecture

```
┌─────────────────────────────────────────┐
│      Llama 3.1 8B (frozen weights)      │
│      + LoRA adapters (trainable)        │
└─────────────────┬───────────────────────┘
                  │
          Hidden States [T, 4096]
                  │
          Read from slot marker positions
                  │
         ┌────────┴────────┐
         ▼                 ▼
    ┌─────────┐      ┌───────────┐
    │NullHead │      │PointerHead│
    │ [4096,2]│      │[4096,4096]│
    └────┬────┘      └─────┬─────┘
         │                 │
         ▼                 ▼
    present/absent    start/end positions
```

### What Gets Trained

| Component | Parameters | Trainable | Purpose |
|-----------|------------|-----------|---------|
| Llama 8B backbone | ~8B | Frozen | The "knowledge" |
| LoRA adapters | ~8M | Yes | Teaches model the input format |
| NullHead | ~8K | Yes | Presence classification |
| PointerHead | ~33M | Yes | Position finding |

LoRA (Low-Rank Adaptation) adds small trainable matrices to the attention layers. This teaches the model to encode slot information at the marker positions without changing the base weights.

All trainable components are trained jointly in a single training run.

## Results

On Llama 3.1 8B Instruct with 1 epoch of training:

| Metric | Value |
|--------|-------|
| Training examples | 8,874 |
| Holdout examples | 986 |
| Presence accuracy | 97.4% |
| **Exact match** | **98.6%** |
| Training time | ~15 minutes (A100) |
| Inference | ~20ms per utterance |

## Contents

```
probe_public/
├── README.md              ← You are here
├── SCIENCE.md             ← Academic background on probing
├── requirements.txt       ← Python dependencies
│
├── data/
│   ├── DATA_CARD.md                   # Data provenance and format
│   ├── compiled_clean_shuffled.jsonl  # 9,860 training examples
│   └── function_schemas.json          # 45 function slot definitions
│
├── src/
│   ├── probes.py          # NullHead and PointerHead classes
│   ├── dataset.py         # Data loading and tokenization
│   └── train.py           # Training script
│
├── modal/
│   └── MODAL_GUIDE.md     # Cloud GPU setup (Modal.com)
│
└── appendix/
    ├── VLLM_DEPLOYMENT.md # Production deployment notes
    ├── QUANTIZATION.md    # 4-bit quantization results
    └── MULTI_MODEL.md     # Results on Qwen, OLMo, etc.
```

## Requirements

- **Python**: 3.10 or 3.11
- **GPU**: CUDA-capable, 24GB+ VRAM (A10G, A100, RTX 3090/4090)
- **Llama access**: Approved access to `meta-llama/Llama-3.1-8B-Instruct` on HuggingFace

### Install

```bash
pip install -r requirements.txt
```

Dependencies: `torch>=2.0.0`, `transformers>=4.40.0`, `peft>=0.10.0`, `accelerate>=0.27.0`

### HuggingFace Setup

```bash
huggingface-cli login
# You must have accepted Meta's license at:
# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

## Quick Start

**Local (with GPU):**
```bash
cd src
python train.py --data ../data --epochs 1
```

**Cloud (~$0.50 on Modal):**
See [modal/MODAL_GUIDE.md](modal/MODAL_GUIDE.md)

## Training Details

### Data Format

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

Character spans are converted to token spans at training time.

### Loss Function

```python
for each slot in schema:
    query = hidden[slot_marker_position]

    # Is the slot present?
    null_loss = cross_entropy(null_head(query), present_or_absent)

    # Where is it? (only if present)
    if present:
        start_logits, end_logits = pointer_head(query, all_hidden)
        pointer_loss = cross_entropy(start_logits, gold_start)
                     + cross_entropy(end_logits, gold_end)
```

Cross-entropy treats positions as classes. Training pushes the correct position's score up and others down.

### Output Files

```
output/
├── lora_adapter/       # LoRA weights (load with PEFT)
└── probe_heads.pt      # NullHead + PointerHead state dicts
```

## Code Overview

### `src/probes.py`

Defines the two probe heads:

```python
class NullHead(nn.Module):
    # Linear(4096, 2) - binary present/absent classifier

class PointerHead(nn.Module):
    # W_start and W_end matrices [4096, 4096]
    # Bilinear comparison: query @ W @ hidden.T
```

### `src/dataset.py`

Converts raw data to training format:
- Tokenizes utterances
- Converts character spans to token spans
- Builds input sequences with slot markers
- Creates valid_mask (which positions can be pointed to)

### `src/train.py`

Main training script:
1. Load Llama 8B with LoRA
2. Initialize probe heads (random)
3. Training loop: forward → loss → backward → update
4. Evaluate on holdout
5. Save weights

## Beyond This Repo

The appendix documents additional experiments:
- **vLLM deployment**: Hidden state normalization issues and fixes
- **AWQ quantization**: 4-bit inference with 98.0% accuracy (vs 98.6%)
- **Other models**: Qwen 96.1%, OLMo 98.3%, Nemotron 98.1%

These are optional reading - the core approach is fully contained in this README and the src/ code.

## License

Data derived from Salesforce xLAM-60k, licensed **CC-BY-NC-4.0**.

**Research use only. Not for commercial applications.**

## Citation

- xLAM dataset: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
- Probing papers: See [SCIENCE.md](SCIENCE.md)
