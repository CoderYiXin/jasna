from __future__ import annotations

import torch

from jasna.mosaic import Detections


class FramesRestorer:
    def __init__(self, *, clip_len: int, alpha: float = 0.3) -> None:
        self.clip_len = int(clip_len)
        self.alpha = float(alpha)

    def restore(self, frames_uint8_bchw: torch.Tensor, detections: Detections) -> torch.Tensor:
        keep_k = torch.isfinite(detections.scores)
        union_masks = (detections.masks & keep_k[:, :, None, None]).any(dim=1)  # (B, H, W)

        frames_f16 = frames_uint8_bchw.to(dtype=torch.float16, device=frames_uint8_bchw.device)
        red = torch.tensor([255.0, 0.0, 0.0], device=frames_uint8_bchw.device, dtype=torch.float16)[:, None, None]

        blended = torch.where(
            union_masks[:, None, :, :],
            frames_f16 * (1.0 - self.alpha) + red * self.alpha,
            frames_f16,
        )
        return blended.clamp(0, 255).to(torch.uint8)

