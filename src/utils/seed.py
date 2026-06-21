from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed all RNGs. If `deterministic`, also enable cuDNN deterministic mode
    (slower but bit-exact reproducible across runs with the same seed).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # cuBLAS workspace must be set BEFORE the first cuBLAS call.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # `warn_only=True` so we get a warning instead of an error if some op
        # has no deterministic kernel (e.g. scatter on CUDA).
        torch.use_deterministic_algorithms(True, warn_only=True)
