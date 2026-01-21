# Probe-Based Slot Extraction

This project extracts slot values from natural language by attaching lightweight classifiers ("probes") to a large language model's internal representations, rather than having the model generate text output.

Given an utterance like "Book a flight from Seattle to Denver tomorrow", the system identifies that "Seattle" fills the `origin` slot, "Denver" fills the `destination` slot, and "tomorrow" fills the `date` slot - all in a single forward pass through the model.

For background on probing as a technique, see [SCIENCE.md](SCIENCE.md). For details on this specific implementation, see [PROBE.md](PROBE.md).

## Contents

```
probe_public/
│
├── README.md          ← You are here
├── SCIENCE.md         ← General background on probing techniques
├── PROBE.md           ← This project's implementation details
├── requirements.txt   ← Python dependencies
│
├── data/
│   ├── DATA_CARD.md                   # Data provenance and format
│   ├── compiled_clean_shuffled.jsonl  # Training data (9,860 examples)
│   └── function_schemas.json          # Slot definitions (45 functions)
│
├── src/
│   ├── dataset.py     # Data loading and tokenization
│   ├── train.py       # Training script
│   └── probes.py      # Probe head definitions (NullHead, PointerHead)
│
├── modal/
│   └── MODAL_GUIDE.md # Instructions for running on Modal.com
│
└── appendix/
    ├── VLLM_DEPLOYMENT.md  # Notes on vLLM inference deployment
    ├── QUANTIZATION.md     # AWQ/INT4 quantization results
    └── MULTI_MODEL.md      # Results on other models (Qwen, OLMo, etc.)
```

## Requirements

- **Python**: 3.10 or 3.11
- **GPU**: CUDA-capable, 24GB+ VRAM recommended (A10G, A100, RTX 3090/4090)
- **Llama access**: Requires approved access to `meta-llama/Llama-3.1-8B-Instruct` on HuggingFace

### Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies:
- `torch>=2.0.0` (with CUDA)
- `transformers>=4.40.0`
- `peft>=0.10.0`
- `accelerate>=0.27.0`
- `safetensors>=0.4.0`

### HuggingFace Setup

```bash
# Login to access Llama
huggingface-cli login

# You must have accepted Meta's license at:
# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

## Quick Start

**Local (if you have a GPU):**
```bash
cd src
python train.py --data ../data --epochs 1
```

**Cloud (Modal.com, ~$0.50):**
See [modal/MODAL_GUIDE.md](modal/MODAL_GUIDE.md)

For details on the approach, see [PROBE.md](PROBE.md).

## Results

On Llama 3.1 8B Instruct with 1 epoch of training:
- **Slot presence accuracy**: 97.4%
- **Exact value match**: 98.6%
- **Inference latency**: ~20ms per utterance

## License

This project uses data derived from the Salesforce xLAM Function Calling 60K dataset, which is licensed under **CC-BY-NC-4.0** (Creative Commons Attribution-NonCommercial).

**Research use only.** Not for commercial applications.

## Citation

If you use this work, please cite:
- The xLAM dataset: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
- Relevant probing papers listed in [SCIENCE.md](SCIENCE.md)
