from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Detections:
    scores: np.ndarray  # (B, K) on CPU
    boxes_xyxy: np.ndarray  # (B, K, 4) in pixels, original frame space, on CPU
    masks: torch.Tensor  # (B, K, H, W) bool, original frame space, on GPU

