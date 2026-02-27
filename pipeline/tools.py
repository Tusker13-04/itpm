from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
import torch
from PIL import Image
from groundingdino.util.inference import load_image, predict
from langchain_core.tools import tool

from models.gdino import get_gdino
from models.sam2 import get_sam2
from models.clip_model import get_clip


@tool
def run_grounding_dino(
    image_path: str,
    prompt: str,
    box_thresh: float = 0.35,
    text_thresh: float = 0.25,
) -> Dict[str, Any]:
    """
    Detects objects matching a text prompt using Grounding DINO.

    Args:
        image_path: Path to input image.
        prompt: Dot-separated classes, e.g. "person . laptop . bag".
        box_thresh: Minimum box confidence threshold.
        text_thresh: Text score threshold.

    Returns:
        Dict with keys:
            - boxes   : list of normalized [cx, cy, w, h] (GroundingDINO default)
            - phrases : list of detected text phrases
            - logits  : list of confidence scores
    """
    model = get_gdino()

    img_src, img_tensor = load_image(image_path)
    boxes, logits, phrases = predict(
        model=model,
        image=img_tensor,
        caption=prompt,
        box_threshold=box_thresh,
        text_threshold=text_thresh,
    )

    return {
        "boxes": boxes.tolist(),
        "phrases": list(phrases),
        "logits": logits.tolist(),
    }


@tool
def run_sam2(image_path: str, boxes: List[List[float]]) -> Dict[str, Any]:
    """
    Generates pixel-level masks for each bounding box using SAM2.

    Args:
        image_path: Path to input image.
        boxes: List of normalized [cx, cy, w, h] boxes from GroundingDINO.

    Returns:
        Dict with keys:
            - masks       : list of binary masks (as lists)
            - mask_scores : list of mask scores
    """
    predictor = get_sam2()

    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]

    # Convert normalized [cx, cy, w, h] to absolute [x1, y1, x2, y2]
    abs_boxes = []
    for b in boxes:
        cx, cy, bw, bh = b[0], b[1], b[2], b[3]
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        abs_boxes.append([x1, y1, x2, y2])

    predictor.set_image(img)
    masks, scores, _ = predictor.predict(
        box=np.array(abs_boxes),
        multimask_output=False,
    )

    return {"masks": masks.tolist(), "mask_scores": scores.tolist()}


@tool
def run_clip_rerank(
    image_path: str,
    boxes: List[List[float]],
    phrases: List[str],
) -> Dict[str, Any]:
    """
    Re-scores boxes via CLIP text-image cosine similarity.

    Crops each box region for precision — mimics YOLO-World RepVL-PAN.

    Returns:
        Dict with key:
            - clip_scores: list of scores in [0, 1]
    """
    model, processor = get_clip()
    device = next(model.parameters()).device
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    scores: List[float] = []

    for box, phrase in zip(boxes, phrases):
        # Grounding DINO outputs normalized [center_x, center_y, width, height]
        cx, cy, bw, bh = box[0], box[1], box[2], box[3]
        
        # Convert to absolute [x1, y1, x2, y2]
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # Ensure valid crop boundaries for PIL (x1 < x2 and y1 < y2)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        crop = img.crop((x1, y1, x2, y2))

        inputs = processor(
            text=[phrase],
            images=crop,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            s = model(**inputs).logits_per_image.sigmoid().item()

        scores.append(float(s))

    return {"clip_scores": scores}
