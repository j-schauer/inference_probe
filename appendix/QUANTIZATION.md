# Quantization Notes

This appendix documents AWQ (4-bit) quantization experiments.

## Key Finding

LoRA adapters and probe heads trained on FP16 transfer directly to AWQ **with no retraining**.

| Model | Exact Match | VRAM |
|-------|-------------|------|
| FP16 | 98.6% | 14.3 GB |
| AWQ 4-bit | 98.0% | 4.7 GB |

0.6% accuracy drop for 3x memory reduction.

## Why It Works

1. **LoRA captures direction, not magnitude** - The low-rank update learns which direction to move weights, not absolute values
2. **AWQ preserves distribution** - Activation-aware quantization maintains hidden state characteristics
3. **Probes are scale-tolerant** - With normalization disabled, relative magnitudes are preserved

## Model Pairing Requirement

**Critical**: FP16 and AWQ models must be from the same source.

| FP16 Training Model | AWQ Inference Model | Works? |
|---------------------|---------------------|--------|
| `meta-llama/Llama-3.1-8B-Instruct` | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | Yes |
| `NousResearch/Hermes-2-Pro-Llama-3-8B` | `casperhansen/llama-3-8b-instruct-awq` | **No** |

Using the wrong AWQ model gave 81.5% accuracy - the models are different despite similar names.

## Performance

| Metric | FP16 | AWQ |
|--------|------|-----|
| Accuracy | 98.6% | 98.0% |
| VRAM | 14.3 GB | 4.7 GB |
| Latency (batch=1) | 21 ms | 56 ms |
| Latency (batch=8) | 35 ms | 42 ms |

AWQ is slower at batch=1 due to dequantization overhead. At larger batches, memory bandwidth becomes the bottleneck and AWQ catches up.

## vLLM AWQ Configuration

```python
llm = LLM(
    model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    runner="pooling",
    enable_lora=True,
    quantization="awq",
    dtype="float16",
    override_pooler_config=PoolerConfig(normalize=False),
)
```

Same LoRA adapter and probe heads - no changes needed.

## When to Use AWQ

**Use AWQ if:**
- Memory constrained (e.g., consumer GPU)
- Batching multiple requests
- 0.6% accuracy drop acceptable

**Use FP16 if:**
- Single-request latency critical
- Accuracy paramount
- Sufficient VRAM available

## Other Quantization Methods

We only tested AWQ. Other options:
- **GPTQ**: Similar to AWQ, may have different accuracy/speed tradeoffs
- **bitsandbytes**: HF-native, may not work with vLLM pooling
- **GGUF**: For llama.cpp, not tested with our pipeline
