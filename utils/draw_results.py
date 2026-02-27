from __future__ import annotations

from typing import List, Dict, Any

import cv2
import numpy as np


def draw_detections(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Draw bounding boxes, labels, and (optionally) masks on an image.

    Args:
        image_bgr: Input image in BGR format (as used by OpenCV).
        detections: List of dicts with keys:
            - label: str
            - box: [x1, y1, x2, y2] in absolute pixels
            - score: float
            - mask: optional 2D mask aligned with the image
        alpha: Opacity for mask overlay.

    Returns:
        A copy of the image with visualizations applied.
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]

    for det in detections:
        label = det.get("label", "obj")
        score = det.get("score", 0.0)
        box = det.get("box", [0, 0, 0, 0])
        mask = det.get("mask")

        x1, y1, x2, y2 = map(int, box)

        # Draw mask overlay if provided
        if mask is not None:
            mask_arr = np.array(mask, dtype=np.uint8)
            if mask_arr.shape[:2] != (h, w):
                # Best-effort resize
                mask_arr = cv2.resize(mask_arr, (w, h), interpolation=cv2.INTER_NEAREST)
            color = (0, 255, 0)
            colored_mask = np.zeros_like(out, dtype=np.uint8)
            colored_mask[mask_arr > 0] = color
            out = cv2.addWeighted(colored_mask, alpha, out, 1 - alpha, 0)

        # Draw bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw label + score
        text = f"{label} {score:.2f}"
        cv2.putText(
            out,
            text,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            text,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return out

