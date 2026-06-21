"""Pre-compute Stage I text features for every utterance in the IEMOCAP manifests.

Stage II reuses these features as input — RoBERTa-large is frozen here, so it's
~10× faster to extract once and cache rather than running roberta forward each
training step.

Output: a single .pt file holding a dict {utt_id: tensor(1024,)}.

Usage:
    python -m scripts.extract_text_features \
        --stage1-ckpt outputs/iemocap_robertaft_lr2e5/best \
        --manifest-dir data/manifests/iemocap \
        --out-pt data/cache/iemocap_text_features.pt
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.models.text_classifier import RobertaTextClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-ckpt", required=True, type=str,
                        help="Folder containing the Stage I best checkpoint "
                             "(produced by src.train, contains HF encoder + classifier.pt).")
    parser.add_argument("--manifest-dir", required=True, type=str)
    parser.add_argument("--out-pt", required=True, type=str)
    parser.add_argument("--num-labels", default=4, type=int,
                        help="Must match the Stage I run (4 for IEMOCAP).")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--max-length", default=128, type=int)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.stage1_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)

    # Load encoder + classifier head (we only need get_features, but we re-init
    # the full module so the saved state_dict loads cleanly).
    model = RobertaTextClassifier(pretrained=str(ckpt_dir), num_labels=int(args.num_labels))
    full_ckpt = ckpt_dir / "classifier.pt"
    if full_ckpt.exists():
        state = torch.load(full_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"], strict=False)
        print(f"Loaded classifier.pt from epoch {state.get('epoch')}, score {state.get('score')}")
    else:
        print(f"[warn] {full_ckpt} not found — using bare HF weights from {ckpt_dir}")
    model = model.to(device).eval()

    # Read all utterances across splits
    manifest_dir = Path(args.manifest_dir)
    rows = []
    seen_ids = set()
    for split in ("train", "val", "test"):
        csv_path = manifest_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                uid = r.get("utt_id")
                if not uid or uid in seen_ids:
                    continue
                seen_ids.add(uid)
                rows.append((uid, str(r.get("text", ""))))
    print(f"Extracting features for {len(rows)} utterances across splits...")

    out_pt = Path(args.out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)

    features = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(rows), args.batch_size), desc="extract"):
            batch = rows[i : i + args.batch_size]
            ids = [b[0] for b in batch]
            texts = [b[1] for b in batch]
            enc = tokenizer(
                texts, padding=True, truncation=True,
                max_length=args.max_length, return_tensors="pt",
            )
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            feats = model.get_features(enc["input_ids"], enc["attention_mask"])  # (B, 1024)
            feats = feats.detach().cpu().to(torch.float32)
            for uid, f in zip(ids, feats):
                features[uid] = f.clone()

    print(f"Saving {len(features)} features (dim={next(iter(features.values())).shape}) to {out_pt}")
    torch.save(features, out_pt)
    size_mb = out_pt.stat().st_size / (1024 * 1024)
    print(f"Saved {out_pt} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
