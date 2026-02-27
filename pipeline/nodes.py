from __future__ import annotations

import logging
from typing import Dict, Any

from pipeline.state import VisionState
from pipeline.tools import run_grounding_dino, run_sam2, run_clip_rerank

log = logging.getLogger(__name__)

CLIP_THRESHOLD = 0.20


def node_gdino(state: VisionState) -> Dict[str, Any]:
    """Node wrapper around the Grounding DINO tool."""
    log.info("[GDINO] prompt='%s'", state["prompt"])

    try:
        result = run_grounding_dino.invoke(
            {
                "image_path": state["image_path"],
                "prompt": state["prompt"],
            }
        )

        if not result["boxes"]:
            log.warning("[GDINO] No detections")
            return {
                "boxes": [],
                "phrases": [],
                "logits": [],
                "error": "no_detections",
            }

        log.info("[GDINO] %d objects: %s", len(result["boxes"]), result["phrases"])

        return {
            "boxes": result["boxes"],
            "phrases": result["phrases"],
            "logits": result["logits"],
            "error": None,
        }
    except Exception as e:  # pragma: no cover - defensive logging
        log.error("[GDINO] %s", e)
        return {"error": str(e), "boxes": []}


def node_sam2(state: VisionState) -> Dict[str, Any]:
    """Node wrapper around the SAM2 segmentation tool."""
    num_boxes = len(state.get("boxes") or [])
    log.info("[SAM2] %d boxes", num_boxes)

    try:
        result = run_sam2.invoke(
            {
                "image_path": state["image_path"],
                "boxes": state["boxes"],
            }
        )

        log.info("[SAM2] %d masks generated", len(result["masks"]))
        return {"masks": result["masks"]}
    except Exception as e:  # pragma: no cover - defensive logging
        log.error("[SAM2] %s", e)
        return {"error": str(e), "masks": []}


def node_clip(state: VisionState) -> Dict[str, Any]:
    """Node wrapper around CLIP re-ranking."""
    num_boxes = len(state.get("boxes") or [])
    log.info("[CLIP] scoring %d regions", num_boxes)

    try:
        result = run_clip_rerank.invoke(
            {
                "image_path": state["image_path"],
                "boxes": state["boxes"],
                "phrases": state["phrases"],
            }
        )

        log.info("[CLIP] scores=%s", result["clip_scores"])
        return {"clip_scores": result["clip_scores"]}
    except Exception as e:  # pragma: no cover - defensive logging
        log.error("[CLIP] %s", e)
        return {"error": str(e), "clip_scores": []}


def node_filter(state: VisionState) -> Dict[str, Any]:
    """
    Filters detections using CLIP scores and packs final results.
    """
    log.info("[FILTER] threshold=%.3f", CLIP_THRESHOLD)

    boxes = state.get("boxes") or []
    phrases = state.get("phrases") or []
    masks = state.get("masks") or []
    clip_scores = state.get("clip_scores") or []

    final = [
        {
            "label": phrase,
            "box": box,
            "mask": mask,
            "score": round(float(score), 4),
        }
        for box, phrase, mask, score in zip(boxes, phrases, masks, clip_scores)
        if score >= CLIP_THRESHOLD
    ]

    log.info("[FILTER] kept %d/%d", len(final), len(boxes))
    return {"final": final}

