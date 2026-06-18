"""Tiny logger that fans out to stdout + TensorBoard + (optional) WandB."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter

try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except Exception:  # pragma: no cover
    _WANDB_AVAILABLE = False


class RunLogger:
    def __init__(
        self,
        output_dir: str | Path,
        run_name: str,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.tb = SummaryWriter(log_dir=str(self.output_dir / "tb"))

        self.logger = logging.getLogger(run_name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                  datefmt="%H:%M:%S")
            )
            self.logger.addHandler(handler)
            file_handler = logging.FileHandler(self.output_dir / "train.log", encoding="utf-8")
            file_handler.setFormatter(handler.formatter)
            self.logger.addHandler(file_handler)

        self.metrics_path = self.output_dir / "metrics.jsonl"

        self.use_wandb = bool(use_wandb and _WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY"))
        if use_wandb and not self.use_wandb:
            self.logger.warning(
                "use_wandb=True but WANDB_API_KEY missing or wandb not installed — "
                "falling back to TensorBoard only."
            )
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=run_name,
                dir=str(self.output_dir),
                config=wandb_config or {},
                reinit=True,
            )

    # --- helpers --------------------------------------------------------
    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def log_scalars(self, scalars: Dict[str, float], step: int, prefix: str = "") -> None:
        for k, v in scalars.items():
            tag = f"{prefix}/{k}" if prefix else k
            self.tb.add_scalar(tag, v, step)
        if self.use_wandb:
            wandb.log({f"{prefix}/{k}" if prefix else k: v for k, v in scalars.items()}, step=step)
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "prefix": prefix, **scalars}) + "\n")

    def close(self) -> None:
        self.tb.flush()
        self.tb.close()
        if self.use_wandb:
            wandb.finish()
