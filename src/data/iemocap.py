"""IEMOCAP text manifest builder.

Reads the official `IEMOCAP_full_release/` directory and produces a unified
manifest of (utt_id, session, dialogue_id, text, emotion, split) rows.

Paper protocol (Sec. IV-A):
    - 4-way classification: angry / happy(+excited) / sad / neutral
    - Session 5 -> test, Session 1 -> val, Sessions 2-4 -> train
    - 5531 utterances total
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

# ----------------------------------------------------------------------------
# Regex for IEMOCAP files
# ----------------------------------------------------------------------------
# Transcription line:
#   Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me.
_TRANS_RE = re.compile(
    r"^(?P<utt_id>Ses\d{2}[FM]_\w+?_[FM]\d{3,4})\s+\[[^\]]+\]:\s*(?P<text>.*)$"
)

# Emo evaluation summary line:
#   [6.2901 - 8.2357]\tSes01F_impro01_F000\tneu\t[2.5000, 2.5000, 2.5000]
_EMO_RE = re.compile(
    r"^\[[\d\.\s\-]+\]\s+(?P<utt_id>Ses\d{2}[FM]_\w+?_[FM]\d{3,4})\s+(?P<emo>\w+)\s+"
)


@dataclass
class IemocapRow:
    utt_id: str
    session: int
    dialogue_id: str
    text: str
    emotion: str  # raw IEMOCAP label, e.g. ang/hap/exc/sad/neu/fru/xxx


def _parse_transcription(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _TRANS_RE.match(line.strip())
            if m:
                out[m.group("utt_id")] = m.group("text").strip()
    return out


def _parse_emotions(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _EMO_RE.match(line.strip())
            if m:
                out[m.group("utt_id")] = m.group("emo").lower()
    return out


def load_iemocap_rows(root: str | Path) -> List[IemocapRow]:
    """Walk all 5 sessions and return every utterance with a transcript+label."""
    root = Path(root)
    rows: List[IemocapRow] = []
    for sess_idx in range(1, 6):
        session_dir = root / f"Session{sess_idx}" / "dialog"
        trans_dir = session_dir / "transcriptions"
        emo_dir = session_dir / "EmoEvaluation"
        if not trans_dir.is_dir() or not emo_dir.is_dir():
            raise FileNotFoundError(
                f"Missing IEMOCAP layout under {session_dir}. "
                "Expecting transcriptions/ and EmoEvaluation/."
            )
        for emo_path in sorted(emo_dir.glob("Ses*.txt")):
            dialogue_id = emo_path.stem  # e.g. Ses01F_impro01
            trans_path = trans_dir / f"{dialogue_id}.txt"
            if not trans_path.exists():
                continue
            texts = _parse_transcription(trans_path)
            emotions = _parse_emotions(emo_path)
            for utt_id, emo in emotions.items():
                if utt_id not in texts:
                    continue
                rows.append(
                    IemocapRow(
                        utt_id=utt_id,
                        session=sess_idx,
                        dialogue_id=dialogue_id,
                        text=texts[utt_id],
                        emotion=emo,
                    )
                )
    return rows


def build_manifest(
    root: str | Path,
    label_map: Dict[str, int],
    train_sessions: Iterable[int],
    val_sessions: Iterable[int],
    test_sessions: Iterable[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build train/val/test DataFrames following the paper's protocol.

    Only utterances whose raw emotion appears in ``label_map`` survive (the
    paper keeps {ang, hap, exc, sad, neu}; everything else — fru, sur, fea,
    dis, oth, xxx — is dropped).
    """
    rows = load_iemocap_rows(root)
    train_s, val_s, test_s = set(train_sessions), set(val_sessions), set(test_sessions)

    keep = []
    for r in rows:
        if r.emotion not in label_map:
            continue
        if r.session in train_s:
            split = "train"
        elif r.session in val_s:
            split = "val"
        elif r.session in test_s:
            split = "test"
        else:
            continue
        keep.append(
            dict(
                utt_id=r.utt_id,
                dialogue_id=r.dialogue_id,
                session=r.session,
                text=r.text,
                raw_emotion=r.emotion,
                label=label_map[r.emotion],
                split=split,
            )
        )

    if not keep:
        raise RuntimeError(
            "No IEMOCAP utterances passed the label filter. Check `label_map` "
            "in your config and the dataset path."
        )

    df = pd.DataFrame(keep)
    return (
        df[df.split == "train"].reset_index(drop=True),
        df[df.split == "val"].reset_index(drop=True),
        df[df.split == "test"].reset_index(drop=True),
    )


def write_manifests(
    root: str | Path,
    manifest_dir: str | Path,
    label_map: Dict[str, int],
    train_sessions: Iterable[int],
    val_sessions: Iterable[int],
    test_sessions: Iterable[int],
) -> Dict[str, Path]:
    """Materialize train/val/test CSVs and return the paths."""
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = build_manifest(
        root, label_map, train_sessions, val_sessions, test_sessions
    )
    paths = {}
    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = manifest_dir / f"{split}.csv"
        df.to_csv(out, index=False)
        paths[split] = out
    return paths
