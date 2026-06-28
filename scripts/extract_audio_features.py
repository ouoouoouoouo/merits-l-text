"""Pre-compute utterance-level audio embeddings for IEMOCAP with WavLM-base.

This is the closest off-the-shelf substitute for CARE: the MERITS-L paper's
CARE encoder is built on a frozen WavLM-base feature extractor + transformer
layers initialized from WavLM-base. We skip CARE's PASE+/RoBERTa-mean dual
supervision and just use the raw WavLM-base features (matching CARE's "size
class" / 768 hidden dim — fair reproduction).

Output: a .pt file with {utt_id: tensor(768,)} dict (mean-pooled over time).

IEMOCAP audio layout (auto-derived from utt_id):
    {root}/Session{N}/sentences/wav/{dialogue_id}/{utt_id}.wav

Usage:
    python -m scripts.extract_audio_features \
        --iemocap-root /home/ouo/dataset/IEMOCAP_full_release \
        --manifest-dir data/manifests/iemocap \
        --out-pt data/cache/iemocap_audio_features_wavlm_base.pt \
        --model microsoft/wavlm-base \
        --batch-size 8
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

# IEMOCAP utt_id like 'Ses01F_impro01_F000' or 'Ses05M_script02_2_M013'
_UTT_RE = re.compile(r"^Ses(\d{2})[FM]_")


def _utt_to_path(iemocap_root: Path, utt_id: str) -> Path | None:
    m = _UTT_RE.match(utt_id)
    if not m:
        return None
    session = int(m.group(1))
    # dialogue_id = everything before the last `_F###` / `_M###`
    parts = utt_id.rsplit("_", 1)
    if len(parts) != 2:
        return None
    dialogue_id, _ = parts
    return (
        iemocap_root
        / f"Session{session}"
        / "sentences"
        / "wav"
        / dialogue_id
        / f"{utt_id}.wav"
    )


def _load_utt_ids(manifest_dir: Path) -> List[str]:
    ids = []
    seen = set()
    for split in ("train", "val", "test"):
        p = manifest_dir / f"{split}.csv"
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                uid = r.get("utt_id")
                if uid and uid not in seen:
                    seen.add(uid)
                    ids.append(uid)
    return ids


def _read_wav(path: Path, target_sr: int = 16000) -> np.ndarray:
    audio, sr = sf.read(str(path))
    if sr != target_sr:
        raise ValueError(f"{path}: expected {target_sr} Hz, got {sr} Hz")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix to mono
    return audio.astype(np.float32)


@torch.no_grad()
def _extract_batch(
    model,
    feature_extractor,
    audios: List[np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    """Returns (B, 768) mean-pooled features."""
    inputs = feature_extractor(
        audios,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    out = model(**inputs)
    hidden = out.last_hidden_state                                # (B, T_out, 768)

    # WavLM downsamples ~320× via the CNN feature extractor. Compute the
    # output frame counts so padding doesn't poison the mean.
    input_lens = inputs["attention_mask"].sum(dim=-1)
    out_lens = model._get_feat_extract_output_lengths(input_lens).to(device)
    T_out = hidden.size(1)
    out_mask = torch.arange(T_out, device=device).unsqueeze(0) < out_lens.unsqueeze(1)
    out_mask = out_mask.unsqueeze(-1).float()                     # (B, T_out, 1)

    denom = out_mask.sum(dim=1).clamp(min=1.0)                    # (B, 1)
    pooled = (hidden * out_mask).sum(dim=1) / denom               # (B, 768)
    return pooled.detach().cpu().to(torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iemocap-root", required=True, type=str)
    parser.add_argument("--manifest-dir", required=True, type=str)
    parser.add_argument("--out-pt", required=True, type=str)
    parser.add_argument("--model", default="microsoft/wavlm-base", type=str,
                        help="HuggingFace model id (microsoft/wavlm-base, microsoft/wavlm-large, "
                             "facebook/hubert-base-ls960, etc.). Default matches CARE backbone.")
    parser.add_argument("--batch-size", default=8, type=int,
                        help="Lower this if you hit OOM on long utterances.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    iemocap_root = Path(args.iemocap_root)
    manifest_dir = Path(args.manifest_dir)
    out_pt = Path(args.out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)

    # Resolve audio paths from manifests
    utt_ids = _load_utt_ids(manifest_dir)
    print(f"Found {len(utt_ids)} unique utt_ids across manifests.")
    items: List[Tuple[str, Path]] = []
    missing = 0
    for uid in utt_ids:
        p = _utt_to_path(iemocap_root, uid)
        if p is None or not p.exists():
            missing += 1
            continue
        items.append((uid, p))
    if missing:
        print(f"[warn] {missing} utterances had no .wav file; skipping.")
    print(f"Will extract features for {len(items)} utterances.")

    # Load model
    print(f"Loading {args.model} on {device} ...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    hidden_dim = model.config.hidden_size
    print(f"Hidden dim: {hidden_dim}")

    # Sort by audio length (ish: just sort by file size) to make batches more uniform
    items.sort(key=lambda x: x[1].stat().st_size)

    features = {}
    failed = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(items), args.batch_size), desc="extract"):
            batch = items[i : i + args.batch_size]
            uids = [b[0] for b in batch]
            try:
                audios = [_read_wav(b[1]) for b in batch]
                pooled = _extract_batch(model, feature_extractor, audios, device)
                for uid, feat in zip(uids, pooled):
                    features[uid] = feat.clone()
            except Exception as e:  # noqa: BLE001
                print(f"\n[warn] batch starting at {uids[0]}: {e}")
                failed += len(batch)

    print(f"\nDone. extracted={len(features)} failed={failed}")
    print(f"Saving to {out_pt} ...")
    torch.save(features, out_pt)
    size_mb = out_pt.stat().st_size / (1024 * 1024)
    print(f"Saved {out_pt} ({size_mb:.1f} MB, dim={hidden_dim})")


if __name__ == "__main__":
    main()
