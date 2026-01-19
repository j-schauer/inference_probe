# Multi-Model Experiments

This appendix documents probe training results on models other than Llama 8B.

## Results Summary

| Model | Size | vLLM FP16 | AWQ | Notes |
|-------|------|-----------|-----|-------|
| Llama 3.1 8B Instruct | 8B | **98.6%** | 98.0% | Primary model |
| Qwen 2.5 7B Instruct | 7B | 96.1% | 95.2% | Good alternative |
| OLMo 3-7B Instruct | 7B | 98.3% | N/A | Near Llama quality |
| Nemotron Nano 8B | 8B | 98.1% | N/A | NVIDIA model |
| LFM2 2.6B (Mamba) | 2.6B | FAIL | FAIL | vLLM LoRA incompatible |

## Model-Specific Notes

### Qwen 2.5 7B

- 2.5% lower than Llama
- Faster inference (smaller model)
- Different tokenization - may need data re-processing for best results

### OLMo 3-7B

- Open model from Allen AI
- Nearly matches Llama quality
- Good choice if avoiding Meta models

### Nemotron Nano 8B

- NVIDIA's distilled model
- Similar accuracy to Llama
- May have licensing advantages for commercial use (check license)

### LFM2 (Mamba)

- State-space model (not transformer)
- vLLM LoRA doesn't support Mamba conv layers
- Would need different training approach

## Why Results Vary

1. **Pre-training data** - Models trained on different corpora have different knowledge
2. **Tokenization** - Different tokenizers affect span alignment
3. **Hidden state organization** - Some models may not linearly encode slot information as well
4. **Instruction tuning** - Quality of instruction-following affects probe task

## Recommendation

For this task, **Llama 3.1 8B Instruct** remains the best choice:
- Highest accuracy
- Well-supported in vLLM
- Good LoRA compatibility
- AWQ quantization works

If Llama is unavailable, **OLMo 3-7B** is a strong open alternative.

## Training Details

All models trained with same configuration:
- LoRA: r=16, alpha=32, targets=[q_proj, k_proj, v_proj, o_proj]
- Epochs: 1
- Learning rate: 2e-5
- Same training data

Differences in hidden_size required adjusting probe head dimensions automatically.
