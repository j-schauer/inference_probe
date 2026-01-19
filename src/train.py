#!/usr/bin/env python3
"""
Training script for probe-based slot extraction.

Trains LoRA adapters + probe heads on Llama 8B for slot extraction.

Usage:
    python train.py --data ../data --model meta-llama/Llama-3.1-8B-Instruct
    python train.py --data ../data --epochs 2 --quick  # Quick test with 500 examples

Output:
    output/
    ├── lora_adapter/       # LoRA weights (PEFT format)
    └── probe_heads.pt      # NullHead + PointerHead state dicts
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import ProbeDataset, make_collate_fn
from probes import NullHead, PointerHead


# Default configuration
CONFIG = {
    "model_path": "meta-llama/Llama-3.1-8B-Instruct",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "batch_size": 4,
    "learning_rate": 2e-5,
    "epochs": 1,
    "output_dir": "output",
}


def load_model_with_lora(config):
    """Load base model with LoRA adapters."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print(f"Loading model: {config['model_path']}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
    ).to(device)

    model.config.use_cache = False
    hidden_size = model.config.hidden_size
    print(f"Hidden size: {hidden_size}")

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_targets"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer, hidden_size, device


def get_hidden_states(model, input_ids, attention_mask):
    """
    Extract post-RMSNorm hidden states from the model.

    Uses last_hidden_state explicitly to ensure consistency.
    """
    base = model.base_model.model if hasattr(model, "base_model") else model
    llama = base.model if hasattr(base, "model") else base

    outputs = llama(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        return_dict=True,
        use_cache=False,
    )

    return outputs.last_hidden_state


def compute_loss(model, pointer_head, null_head, batch, device):
    """Compute combined NullHead + PointerHead loss."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    valid_mask = batch["valid_mask"].to(device)

    hidden = get_hidden_states(model, input_ids, attention_mask)

    null_loss = torch.tensor(0.0, device=device)
    ptr_loss = torch.tensor(0.0, device=device)
    n_null = 0
    n_ptr = 0

    batch_size = input_ids.size(0)
    seq_len = input_ids.size(1)

    for b in range(batch_size):
        example_hidden = hidden[b]
        example_valid_mask = valid_mask[b].unsqueeze(0)
        utt_start = batch["utterance_start"][b]

        for slot_name, query_pos in batch["query_positions"][b].items():
            label = batch["labels"][b][slot_name]

            if query_pos < 0 or query_pos >= seq_len:
                continue

            q = example_hidden[query_pos]

            # NullHead loss (always)
            null_logits = null_head(q)
            present_target = torch.tensor(0 if label["present"] else 1, device=device)
            loss_null = F.cross_entropy(null_logits, present_target.unsqueeze(0))
            null_loss = null_loss + loss_null
            n_null += 1

            # PointerHead loss (only if present)
            if label["present"]:
                gold_start = utt_start + label["start"]
                gold_end = utt_start + label["end"]

                if gold_start < 0 or gold_start >= seq_len or gold_end < 0 or gold_end >= seq_len:
                    continue
                if not valid_mask[b, gold_start] or not valid_mask[b, gold_end]:
                    continue

                start_logits, end_logits = pointer_head(
                    q.unsqueeze(0),
                    example_hidden.unsqueeze(0),
                    example_valid_mask
                )

                gold_start_t = torch.tensor(gold_start, device=device)
                gold_end_t = torch.tensor(gold_end, device=device)

                loss_start = F.cross_entropy(start_logits, gold_start_t.unsqueeze(0))
                loss_end = F.cross_entropy(end_logits, gold_end_t.unsqueeze(0))
                ptr_loss = ptr_loss + loss_start + loss_end
                n_ptr += 1

    avg_null = null_loss / max(n_null, 1)
    avg_ptr = ptr_loss / max(2 * n_ptr, 1) if n_ptr > 0 else torch.tensor(0.0, device=device)
    total_loss = avg_null + avg_ptr

    return total_loss, {"null_loss": avg_null.item(), "ptr_loss": avg_ptr.item(), "n_null": n_null, "n_ptr": n_ptr}


def evaluate(model, pointer_head, null_head, dataset, indices, tokenizer, device):
    """Evaluate on holdout set."""
    model.eval()
    pointer_head.eval()
    null_head.eval()

    correct_presence = 0
    total_presence = 0
    correct_exact = 0
    total_present = 0

    subset = Subset(dataset, indices)
    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    loader = DataLoader(subset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            hidden = get_hidden_states(model, input_ids, attention_mask)
            seq_len = input_ids.size(1)

            for b in range(input_ids.size(0)):
                for slot_name, query_pos in batch["query_positions"][b].items():
                    label = batch["labels"][b][slot_name]

                    if query_pos < 0 or query_pos >= seq_len:
                        continue

                    q = hidden[b, query_pos]

                    null_logits = null_head(q)
                    pred_present = null_logits.argmax().item() == 0
                    gold_present = label["present"]

                    total_presence += 1
                    if pred_present == gold_present:
                        correct_presence += 1

                    if gold_present:
                        total_present += 1
                        if pred_present:
                            start_logits, end_logits = pointer_head(
                                q.unsqueeze(0),
                                hidden[b:b+1],
                                valid_mask[b:b+1]
                            )
                            start_idx = start_logits.argmax().item()
                            end_idx = end_logits.argmax().item()
                            if end_idx < start_idx:
                                end_idx = start_idx

                            utt_start = batch["utterance_start"][b]
                            gold_start = utt_start + label["start"]
                            gold_end = utt_start + label["end"]

                            pred_tokens = input_ids[b, start_idx:end_idx+1]
                            gold_tokens = input_ids[b, gold_start:gold_end+1]

                            pred_value = tokenizer.decode(pred_tokens).strip()
                            gold_value = tokenizer.decode(gold_tokens).strip()

                            if pred_value == gold_value:
                                correct_exact += 1

    model.train()
    pointer_head.train()
    null_head.train()

    return {
        "presence_acc": correct_presence / max(total_presence, 1),
        "exact_match": correct_exact / max(total_present, 1),
    }


def save_outputs(model, pointer_head, null_head, config):
    """Save LoRA adapter and probe heads."""
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    lora_dir = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(lora_dir)
    print(f"Saved LoRA adapter to {lora_dir}")

    probe_path = os.path.join(output_dir, "probe_heads.pt")
    torch.save({
        "pointer_head": pointer_head.state_dict(),
        "null_head": null_head.state_dict(),
    }, probe_path)
    print(f"Saved probe heads to {probe_path}")


def main():
    parser = argparse.ArgumentParser(description="Train probe-based slot extraction")
    parser.add_argument("--data", type=str, required=True, help="Path to data directory")
    parser.add_argument("--model", type=str, default=CONFIG["model_path"], help="Base model path")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"], help="Number of epochs")
    parser.add_argument("--output", type=str, default=CONFIG["output_dir"], help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick test with 500 examples")
    args = parser.parse_args()

    config = CONFIG.copy()
    config["model_path"] = args.model
    config["epochs"] = args.epochs
    config["output_dir"] = args.output

    print("=" * 60)
    print("PROBE-BASED SLOT EXTRACTION TRAINING")
    print("=" * 60)

    model, tokenizer, hidden_size, device = load_model_with_lora(config)

    pointer_head = PointerHead(hidden_size).to(device)
    null_head = NullHead(hidden_size).to(device)

    compiled_file = os.path.join(args.data, "compiled_clean_shuffled.jsonl")
    schema_file = os.path.join(args.data, "function_schemas.json")
    dataset = ProbeDataset(compiled_file, schema_file, tokenizer)

    n_total = len(dataset)
    if args.quick:
        train_indices = list(range(500))
        holdout_indices = list(range(500, 600))
        print("QUICK MODE: 500 train, 100 holdout")
    else:
        n_train = int(n_total * 0.9)
        train_indices = list(range(n_train))
        holdout_indices = list(range(n_train, n_total))

    print(f"Train: {len(train_indices)}, Holdout: {len(holdout_indices)}")

    train_subset = Subset(dataset, train_indices)
    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    params = trainable_params + list(pointer_head.parameters()) + list(null_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=config["learning_rate"])

    print("\n=== BEFORE TRAINING ===")
    metrics = evaluate(model, pointer_head, null_head, dataset, holdout_indices, tokenizer, device)
    print(f"Presence: {metrics['presence_acc']:.1%}, Exact: {metrics['exact_match']:.1%}")

    print(f"\n=== TRAINING ({config['epochs']} epochs) ===")
    for epoch in range(config["epochs"]):
        model.train()
        pointer_head.train()
        null_head.train()

        total_loss = 0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            loss, loss_info = compute_loss(model, pointer_head, null_head, batch, device)

            if not torch.isfinite(loss):
                print(f"[WARN] NaN loss at batch {batch_idx}, skipping")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 50 == 0:
                avg = total_loss / n_batches
                print(f"  Epoch {epoch+1}: batch {batch_idx+1}/{len(train_loader)}, loss={avg:.4f}")

        metrics = evaluate(model, pointer_head, null_head, dataset, holdout_indices, tokenizer, device)
        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, presence={metrics['presence_acc']:.1%}, exact={metrics['exact_match']:.1%}")

    print("\n=== SAVING ===")
    save_outputs(model, pointer_head, null_head, config)

    print("\n=== DONE ===")
    print(f"Final exact match: {metrics['exact_match']:.1%}")


if __name__ == "__main__":
    main()
