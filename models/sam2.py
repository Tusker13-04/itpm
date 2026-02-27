from __future__ import annotations

import os
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

_sam2 = None


def get_sam2():
    """
    Returns a singleton instance of the SAM2 predictor.

    Uses sam2.1_hiera_large.pt by default.
    Fixes Hydra MissingConfigException by passing configs/sam2.1/... relative string.
    """
    global _sam2

    if _sam2 is None:
        checkpoint = os.path.join("weights", "sam2.1_hiera_large.pt")
        # In recent SAM2 versions, you must specify the exact relative path inside the sam2 package
        # e.g., 'configs/sam2.1/sam2.1_hiera_l.yaml'
        # The 'l' stands for large
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        sam2_model = build_sam2(model_cfg, checkpoint, device=device)
        _sam2 = SAM2ImagePredictor(sam2_model)

    return _sam2