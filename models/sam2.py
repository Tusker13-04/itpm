from __future__ import annotations

import os

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

_sam2 = None


def get_sam2():
    """
    Returns a singleton instance of the SAM2 image predictor.

    The weights path follows the project convention:
    - weights/sam2.pt
    """
    global _sam2

    if _sam2 is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = os.path.join("weights", "sam2.pt")

        # The config name comes from the SAM2 docs / project spec.
        sam_model = build_sam2("sam2.1_hiera_large.yaml", checkpoint, device=device)
        _sam2 = SAM2ImagePredictor(sam_model)

    return _sam2

