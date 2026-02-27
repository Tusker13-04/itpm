from __future__ import annotations

import os

import torch
from groundingdino.util.inference import load_model

_gdino = None


def get_gdino():
    """
    Returns a singleton instance of the Grounding DINO model.

    The weights/config paths follow the project convention:
    - weights/gdino_config.py
    - weights/gdino.pth
    """
    global _gdino

    if _gdino is None:
        config_path = os.path.join("weights", "gdino_config.py")
        weights_path = os.path.join("weights", "gdino.pth")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        _gdino = load_model(
            config_path,
            weights_path,
            device=device,
        )

    return _gdino

