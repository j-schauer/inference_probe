# vLLM Deployment Notes

This appendix documents challenges encountered when deploying the trained probes to vLLM for production inference.

**Note**: The main training documentation in PROBE.md uses HuggingFace only. This appendix is for those who want to deploy to vLLM.

## Summary

Deploying HF-trained probes to vLLM required three fixes:
1. Disable hidden state normalization
2. Fix evaluation comparison bug
3. Upgrade vLLM to fix LoRA+pooling issue

After fixes, vLLM achieved **98.6% exact match** - identical to HuggingFace.

## Problem 1: Hidden State Normalization

### Symptom

NullHead predictions flipped between HF and vLLM. PointerHead was fine.

### Cause

vLLM's pooling mode applies L2 normalization by default:

```python
# vLLM source: vllm/model_executor/layers/pooler.py
return F.normalize(pooled_data, p=2, dim=-1)  # Norm becomes 1.0
```

HF returns raw hidden states with norm ~138.

NullHead has a bias term. With normalized inputs (norm=1), the bias dominates tiny logits and flips predictions. PointerHead has no bias, so it's scale-invariant.

### Fix

```python
from vllm.config import PoolerConfig

llm = LLM(
    model=model_path,
    runner="pooling",
    override_pooler_config=PoolerConfig(normalize=False),  # Disable L2 norm
)
```

Add assertion to catch misconfiguration:

```python
mean_norm = hidden.norm(dim=-1).mean().item()
assert mean_norm > 10.0, f"Normalization not disabled: {mean_norm}"
```

## Problem 2: Evaluation Comparison Bug

### Symptom

Initial testing showed HF=98.8% but vLLM=91.8% - a 7% gap.

### Cause

Different gold value comparison:

```python
# HF evaluation (correct)
gold_tokens = input_ids[gold_start:gold_end+1]
gold_value = tokenizer.decode(gold_tokens)

# vLLM evaluation (wrong)
gold_value = label["value"]  # Raw JSON string
```

Tokenization isn't perfectly reversible. Example:
- JSON: `"-1, -2"`
- Decoded from tokens: `"(-1, -2"` (tokenizer included preceding paren)

### Fix

Both evaluations decode gold from token positions:

```python
gold_tokens = input_ids[gold_start:gold_end+1]
gold_value = tokenizer.decode(gold_tokens).strip()
```

Result: vLLM jumped from 91.8% to 98.5%.

## Problem 3: LoRA + Pooling Bug

### Symptom

LoRA adapters silently not applied in vLLM pooling mode (v0.13.0).

### Cause

Module name mismatch:
- PEFT saves: `base_model.model.layers.0.self_attn.q_proj.lora_A.weight`
- vLLM pooling looks for: `model.layers.0.self_attn.q_proj`

vLLM strips `base_model.model.` but pooling models have `model.` prefix, so lookup fails. No error, no warning - silent failure.

### Fix

Upgrade to vLLM version with PR #14935 (merged March 2025).

Or verify LoRA application:

```python
out_base = llm.encode(ids, lora_request=None)
out_lora = llm.encode(ids, lora_request=lora_req)
diff = (out_base - out_lora).abs().mean()
assert diff > 0.01, "LoRA not applied"
```

## Final vLLM Configuration

```python
from vllm import LLM
from vllm.config import PoolerConfig
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    runner="pooling",
    enable_lora=True,
    max_lora_rank=64,
    max_model_len=2048,
    override_pooler_config=PoolerConfig(normalize=False),
    dtype="bfloat16",
)

lora_request = LoRARequest("probe", 1, "path/to/lora_adapter")

outputs = llm.encode(
    [{"prompt_token_ids": input_ids}],
    pooling_task="token_embed",
    lora_request=lora_request,
)
```

## Results

| Runtime | Exact Match |
|---------|-------------|
| HF BF16 | 98.6% |
| vLLM FP16 (after fixes) | 98.6% |
| vLLM AWQ 4-bit | 98.0% |

## Lessons

1. **Add assertions** - Catch misconfigurations immediately
2. **Compare apples to apples** - Decode gold from same source
3. **Test LoRA application** - Don't trust silent success
4. **NullHead is fragile** - Consider removing bias for deployment
