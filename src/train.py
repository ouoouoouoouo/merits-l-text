"""Generic config-driven trainer for the text branch (Stage I & pretrain).

Usage:
    python -m src.train --config configs/iemocap_text.yaml
    python -m src.train --config configs/meld_text.yaml
    python -m src.train --config configs/mosi_text.yaml
    python -m src.train --config configs/pretrain_msp.yaml

The same loop covers the MSP-PODCAST pre-training step and the downstream
Stage I fine-tunes — only the dataset module and config differ.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .data import build_dataloaders
from .data.dataset import compute_class_weights
from .models.text_classifier import build_model, build_tokenizer
from .utils.config import AttrDict, load_config
from .utils.logging import RunLogger
from .utils.metrics import compute_metrics, detailed_report
from .utils.seed import set_seed


# ---------------------------------------------------------------------------
# Manifest construction (dispatch per dataset name)
# ---------------------------------------------------------------------------
def ensure_manifests(cfg: AttrDict) -> Path:
    """Build per-split CSV manifests if missing, return the manifest dir."""
    ds = cfg.dataset
    manifest_dir = Path(ds.manifest_dir)

    expected = {"train.csv", "val.csv", "test.csv"}
    if ds.name == "msp_podcast_pseudo":
        expected = {"train.csv", "val.csv"}
    if all((manifest_dir / fname).exists() for fname in expected):
        return manifest_dir

    print(f"[manifests] Building under {manifest_dir} ...")
    if ds.name == "iemocap":
        from .data import iemocap
        iemocap.write_manifests(
            root=ds.root,
            manifest_dir=manifest_dir,
            label_map=dict(ds.label_map),
            train_sessions=list(ds.train_sessions),
            val_sessions=list(ds.val_sessions),
            test_sessions=list(ds.test_sessions),
        )
    elif ds.name == "meld":
        from .data import meld
        meld.write_manifests(
            root=ds.root,
            manifest_dir=manifest_dir,
            csv_train=ds.csv_train,
            csv_val=ds.csv_val,
            csv_test=ds.csv_test,
            text_column=ds.text_column,
            label_column=ds.label_column,
            label_map=dict(ds.label_map),
        )
    elif ds.name == "mosi":
        from .data import mosi
        if not ds.csv_path:
            raise ValueError(
                "configs/mosi_text.yaml: `dataset.csv_path` is null. "
                "Run scripts/preprocess_mosi.py first, or point csv_path at "
                "your prepared CMU-MOSI CSV."
            )
        mosi.write_manifests(
            csv_path=ds.csv_path,
            manifest_dir=manifest_dir,
            text_column=ds.text_column,
            score_column=ds.score_column,
            id_column=ds.id_column,
            train_monologues=list(ds.train_monologues),
            val_monologues=list(ds.val_monologues),
            test_monologues=list(ds.test_monologues),
        )
    elif ds.name == "msp_podcast_pseudo":
        from .data import msp_podcast
        msp_podcast.write_manifests(
            manifest_path=ds.manifest_path,
            manifest_dir=manifest_dir,
            label_map=dict(ds.label_map),
            val_fraction=float(ds.val_fraction),
            seed=int(cfg.seed),
        )
    else:
        raise ValueError(f"Unknown dataset.name = {ds.name}")
    return manifest_dir


# ---------------------------------------------------------------------------
# Train / eval steps
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, label_names) -> Dict:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    losses: List[float] = []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        losses.append(out["loss"].item())
        preds = out["logits"].argmax(dim=-1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["labels"].detach().cpu().numpy().tolist())
    metrics = compute_metrics(all_labels, all_preds, label_names=label_names)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["_preds"] = all_preds
    metrics["_labels"] = all_labels
    return metrics


def train(cfg: AttrDict) -> None:
    set_seed(int(cfg.seed))
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.snapshot.yaml").write_text(
        json.dumps(dict(cfg), indent=2, default=str), encoding="utf-8"
    )

    runlog = RunLogger(
        output_dir=out_dir,
        run_name=str(cfg.run_name),
        use_wandb=bool(cfg.logging.use_wandb),
        wandb_project=str(cfg.logging.wandb_project),
        wandb_config=dict(cfg),
    )
    runlog.info(f"Output dir: {out_dir.resolve()}")

    # Data
    manifest_dir = ensure_manifests(cfg)
    tokenizer = build_tokenizer(cfg.model.pretrained)
    loaders = build_dataloaders(
        manifest_dir=manifest_dir,
        tokenizer=tokenizer,
        train_batch_size=int(cfg.train.batch_size),
        eval_batch_size=int(cfg.train.eval_batch_size),
        max_length=int(cfg.model.max_length),
        num_workers=int(cfg.train.num_workers),
    )
    runlog.info("Loader sizes: " + ", ".join(f"{k}={len(v.dataset)}" for k, v in loaders.items()))

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    runlog.info(f"Trainable params: {n_params/1e6:.1f}M | device: {device}")

    # Optional class weights (MELD-friendly)
    class_weighted = bool(getattr(cfg.train, "class_weighted_loss", False))
    if class_weighted:
        weights = compute_class_weights(manifest_dir / "train.csv", int(cfg.dataset.num_labels)).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        runlog.info(f"Using class-weighted CE: {weights.tolist()}")
    else:
        loss_fn = nn.CrossEntropyLoss()

    # Optimizer + scheduler
    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": float(cfg.train.weight_decay),
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(grouped, lr=float(cfg.train.learning_rate))
    total_steps = len(loaders["train"]) * int(cfg.train.epochs)
    warmup_steps = int(total_steps * float(cfg.train.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_fp16 = bool(cfg.train.fp16) and device.type == "cuda"
    scaler = GradScaler(device="cuda", enabled=use_fp16)

    label_names = list(cfg.dataset.label_names) if "label_names" in cfg.dataset else None

    best_score = -math.inf
    best_epoch = -1
    patience = int(cfg.train.early_stopping_patience)
    bad_epochs = 0
    log_every = int(cfg.logging.log_every)
    global_step = 0

    best_ckpt_dir = out_dir / "best"

    for epoch in range(int(cfg.train.epochs)):
        model.train()
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch+1}/{cfg.train.epochs}")
        running = []
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_fp16):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=None,  # compute loss outside so class weights work
                )
                loss = loss_fn(out["logits"], batch["labels"])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running.append(loss.item())
            global_step += 1
            if global_step % log_every == 0:
                avg = float(np.mean(running[-log_every:]))
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                runlog.log_scalars(
                    {"loss": avg, "lr": scheduler.get_last_lr()[0]},
                    step=global_step,
                    prefix="train",
                )

        # Validation
        val_metrics = evaluate(model, loaders["val"], device, label_names)
        runlog.log_scalars(
            {k: v for k, v in val_metrics.items() if not k.startswith("_")},
            step=global_step,
            prefix="val",
        )
        runlog.info(
            f"epoch {epoch+1}: val loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} wF1={val_metrics['weighted_f1']:.4f}"
        )

        # Track best
        score = val_metrics[str(cfg.train.save_best_metric)]
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            bad_epochs = 0
            best_ckpt_dir.mkdir(parents=True, exist_ok=True)
            # Save encoder + tokenizer in HF-compatible form so the next stage
            # can `AutoModel.from_pretrained(best_ckpt_dir)` directly.
            model.encoder.save_pretrained(best_ckpt_dir)
            tokenizer.save_pretrained(best_ckpt_dir)
            # And the full classifier weights (encoder + head) for resumption.
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch + 1, "score": score},
                best_ckpt_dir / "classifier.pt",
            )
            runlog.info(f"  -> new best ({cfg.train.save_best_metric}={score:.4f}) saved.")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                runlog.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
                break

    # Test using the best checkpoint
    if "test" in loaders and best_epoch > 0:
        runlog.info(f"Reloading best checkpoint from epoch {best_epoch} for test evaluation.")
        ckpt = torch.load(best_ckpt_dir / "classifier.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics = evaluate(model, loaders["test"], device, label_names)
        runlog.log_scalars(
            {k: v for k, v in test_metrics.items() if not k.startswith("_")},
            step=global_step,
            prefix="test",
        )
        runlog.info(
            f"TEST | acc={test_metrics['accuracy']:.4f} "
            f"weighted_f1={test_metrics['weighted_f1']:.4f} "
            f"macro_f1={test_metrics['macro_f1']:.4f}"
        )
        report = detailed_report(test_metrics["_labels"], test_metrics["_preds"], label_names)
        (out_dir / "test_report.txt").write_text(report, encoding="utf-8")
        runlog.info("\n" + report)

    runlog.info(f"DONE. Best {cfg.train.save_best_metric} = {best_score:.4f} (epoch {best_epoch}).")
    runlog.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="key=value pairs that override config entries (dot.notation). "
             "Example: --override train.epochs=5 train.batch_size=16",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    for ov in args.override:
        key, _, val = ov.partition("=")
        keys = key.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        # Try to parse JSON literal first (so 5 -> int, 1.0e-4 -> float, true -> bool).
        try:
            parsed = json.loads(val)
        except json.JSONDecodeError:
            parsed = val
        node[keys[-1]] = parsed

    train(cfg)


if __name__ == "__main__":
    main()
