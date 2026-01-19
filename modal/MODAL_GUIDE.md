# Running on Modal.com

Modal is a serverless GPU platform. You pay per second of compute, no instances to manage.

## Setup

### 1. Create Account

Go to https://modal.com and sign up. New accounts get $30 free credits.

### 2. Install CLI

```bash
pip install modal
```

### 3. Authenticate

```bash
modal token new
```

This opens a browser for authentication.

## Key Concepts

### Functions

Modal runs Python functions in the cloud. Decorate with `@modal.function()`:

```python
import modal

app = modal.App("my-app")

@app.function(gpu="A10G")
def train():
    # This runs on an A10G GPU in the cloud
    ...
```

### Images

Define the container environment:

```python
image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers",
    "peft",
)

@app.function(image=image, gpu="A10G")
def train():
    ...
```

### Volumes

Persistent storage across function calls:

```python
vol = modal.Volume.from_name("my-volume", create_if_missing=True)

@app.function(volumes={"/data": vol})
def train():
    # /data persists between runs
    ...
```

## GPU Options

| GPU | VRAM | Cost/hr | Use Case |
|-----|------|---------|----------|
| T4 | 16GB | ~$0.60 | Small models, inference |
| A10G | 24GB | ~$1.10 | Medium models, fine-tuning |
| A100-40GB | 40GB | ~$3.00 | Large models |
| A100-80GB | 80GB | ~$4.50 | Very large models |

For Llama 8B training, A10G is sufficient. A100 is faster but more expensive.

## Cost Estimate

Training this probe project:
- **A10G**: ~25 minutes = **~$0.50**
- **A100-40GB**: ~15 minutes = **~$0.75**

You only pay while the function is running.

---

## Project-Specific Setup

### Volume Structure

Create a volume for model weights and data:

```python
vol = modal.Volume.from_name("probe-training", create_if_missing=True)
```

Expected structure after setup:
```
/vol/
├── models/
│   └── llama3-8b-instruct/    # Downloaded base model
├── data/
│   ├── compiled_clean_shuffled.jsonl
│   └── function_schemas.json
└── output/
    ├── lora_adapter/          # Trained LoRA
    └── probe_heads.pt         # Trained probes
```

### Example Training Script

```python
import modal

app = modal.App("probe-training")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers>=4.40.0",
    "peft>=0.10.0",
    "accelerate",
    "safetensors",
)

vol = modal.Volume.from_name("probe-training", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/vol": vol},
    timeout=3600,
)
def train():
    import sys
    sys.path.insert(0, "/vol/src")

    from train import main
    import sys
    sys.argv = [
        "train.py",
        "--data", "/vol/data",
        "--model", "/vol/models/llama3-8b-instruct",
        "--output", "/vol/output",
        "--epochs", "1",
    ]
    main()

    vol.commit()  # Save outputs to volume


@app.function(
    image=image,
    volumes={"/vol": vol},
    timeout=1800,
)
def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "meta-llama/Llama-3.1-8B-Instruct",
        local_dir="/vol/models/llama3-8b-instruct",
        ignore_patterns=["*.gguf", "*.ggml"],
    )

    vol.commit()


@app.local_entrypoint()
def main(action: str = "train"):
    if action == "download":
        download_model.remote()
    elif action == "train":
        train.remote()
    else:
        print(f"Unknown action: {action}")
```

### Running

```bash
# First time: download base model (~15GB, takes a few minutes)
modal run train_modal.py --action download

# Upload your data to volume (one-time)
# Use Modal's volume commands or write an upload function

# Train
modal run train_modal.py --action train
```

### Uploading Data

```python
@app.function(volumes={"/vol": vol})
def upload_data():
    import shutil
    # Copy from local mount or download from URL
    # Then: vol.commit()
```

Or use Modal's CLI:
```bash
modal volume put probe-training ./data /data
modal volume put probe-training ./src /src
```

## Tips

1. **Use volumes** for anything you want to persist (models, data, outputs)
2. **Commit volumes** after writes: `vol.commit()`
3. **Check logs** in Modal dashboard if something fails
4. **Start with A10G** - cheaper, sufficient for 8B models
5. **Use --quick flag** first to verify setup works

## Debugging

```bash
# List your volumes
modal volume list

# See volume contents
modal volume ls probe-training

# Get files from volume
modal volume get probe-training /output/probe_heads.pt ./local_probe_heads.pt
```
