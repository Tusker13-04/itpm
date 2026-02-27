from __future__ import annotations

from typing import Tuple

import torch
from transformers import CLIPModel, CLIPProcessor

_clip = None
_proc = None
_device = None


def get_clip() -> Tuple[CLIPModel, CLIPProcessor]:
    """
    Returns singleton instances of the CLIP model and processor.

    Uses `openai/clip-vit-base-patch32` as specified in the project doc.
    """
    global _clip, _proc, _device

    if _clip is None:
        _clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _clip.to(_device)
        _clip.eval()

    return _clip, _proc

