from __future__ import annotations

import torch

from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
from jasna.tracking.clip_tracker import TrackedClip


class RestorationPipeline:
    def __init__(self, restorer: BasicvsrppMosaicRestorer) -> None:
        self.restorer = restorer

    def restore_clip(
        self, clip: TrackedClip, frames: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Restore a clip by running the restoration model.
        
        clip: TrackedClip with bbox/mask info
        frames: list of (C, H, W) original frames corresponding to clip frames
        
        Returns: list of (C, H_crop, W_crop) restored regions for each frame
        """
        crops: list[torch.Tensor] = []

        for i, frame in enumerate(frames):
            bbox = clip.bboxes[i].astype(int)
            x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), bbox[2], bbox[3]
            x2 = min(frame.shape[2], x2)
            y2 = min(frame.shape[1], y2)
            crop = frame[:, y1:y2, x1:x2]
            crops.append(crop.permute(1, 2, 0))

        restored = self.restorer.restore(crops)
        return [r.permute(2, 0, 1) for r in restored]

