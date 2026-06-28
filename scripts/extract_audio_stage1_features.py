"""Extract Audio Stage I features (post-dense+tanh) for Audio Stage II.

Mirrors scripts/extract_text_features.py but for the audio branch. Reads the
Audio Stage I best checkpoint, runs `get_features()` on each cached WavLM
embedding to produce the S¹_k = (B, hidden_dim) hidden representation that
Stage II's Bi-GRU takes as input.

Output: {utt_id: tensor(hidden_dim,)}.pt

Usage:
    python -m scripts.extract_audio_stage1_features \
        --stage1-ckpt outputs/iemocap_audio_stage1/best/audio_stage1.pt \
        --raw-features data/cache/iemocap_audio_features_wavlm_base.pt \
        --out-pt data/cache/iemocap_audio_features_stage1.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from src.models.audio_classifier import AudioClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-ckpt", required=True, type=str,
                        help="Path to outputs/.../best/audio_stage1.pt")
    parser.add_argument("--raw-features", required=True, type=str,
                        help="Raw WavLM features .pt (from extract_audio_features.py)")
    parser.add_argument("--out-pt", required=True, type=str)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", default=512, type=int)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=True)
    mcfg = ckpt["model_cfg"]
    print(f"Stage I ckpt epoch={ckpt['epoch']}, score={ckpt['score']:.4f}")
    print(f"Model cfg: {mcfg}")

    model = AudioClassifier(
        input_dim=int(mcfg["input_dim"]),
        hidden_dim=int(mcfg.get("hidden_dim", mcfg["input_dim"])),
        num_labels=int(mcfg["num_labels"]),
        dropout=float(mcfg["dropout"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    raw = torch.load(args.raw_features, map_location="cpu", weights_only=True)
    print(f"Loaded {len(raw)} raw features.")

    utt_ids = list(raw.keys())
    out_features = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(utt_ids), args.batch_size), desc="extract"):
            chunk = utt_ids[i : i + args.batch_size]
            feats = torch.stack([raw[u] for u in chunk]).to(device).float()
            hidden = model.get_features(feats)             # (B, hidden_dim)
            hidden = hidden.detach().cpu().to(torch.float32)
            for u, h in zip(chunk, hidden):
                out_features[u] = h.clone()

    out_pt = Path(args.out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_features, out_pt)
    size_mb = out_pt.stat().st_size / (1024 * 1024)
    print(f"Saved {out_pt} ({len(out_features)} features, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
