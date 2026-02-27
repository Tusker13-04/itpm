from __future__ import annotations

import os
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

_sam2 = None


def get_sam2():
    """
    Returns a singleton instance of the SAM2 predictor.
    """
    global _sam2

    if _sam2 is None:
        # Based on your weights folder, the file is just named "sam2.pt"
        checkpoint = os.path.join("weights", "sam2.pt")
        
        # We need to map 'sam2.pt' to the correct architecture config.
        # Assuming the original download script grabbed the base/large model:
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        sam2_model = build_sam2(model_cfg, checkpoint, device=device)
        _sam2 = SAM2ImagePredictor(sam2_model)

    return _sam2