from __future__ import annotations

import logging
from typing import Dict, Any

from langchain_core.messages import AIMessage, HumanMessage

from pipeline.state import VisionState
from pipeline.tools import run_grounding_dino, run_sam2, run_clip_rerank

log = logging.getLogger(__name__)

CLIP_THRESHOLD = 0.20


def process_message(state: VisionState) -> Dict[str, Any]:
    """Extracts prompt and optionally image path from the last user message."""
    messages = state.get("messages", [])
    if not messages:
        return {}
    
    last_message = messages[-1]
    if isinstance(last_message, HumanMessage):
        content = last_message.content
        log.info("[CHAT] Extracted raw message: '%s'", content)
        
        # Simple parsing logic to allow users to provide path in chat
        # Example format: "[C:/path/to/image.jpg] person . car"
        if content.startswith("[") and "]" in content:
            path_end = content.find("]")
            img_path = content[1:path_end].strip()
            prompt_text = content[path_end+1:].strip()
            return {"image_path": img_path, "prompt": prompt_text}
            
        return {"prompt": content}
    return {}


def node_gdino(state: VisionState) -> Dict[str, Any]:
    """Node wrapper around the Grounding DINO tool."""
    image_path = state.get("image_path")
    prompt = state.get("prompt")
    
    log.info("[GDINO] image_path='%s', prompt='%s'", image_path, prompt)
    
    if not image_path:
        return {"error": "Please provide an 'image_path'. You can format your chat message like this: [C:/path/to/image.jpg] person . car", "boxes": []}
    if not prompt:
        return {"error": "Please provide a prompt via chat message.", "boxes": []}

    try:
        result = run_grounding_dino.invoke(
            {
                "image_path": image_path,
                "prompt": prompt,
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
    except Exception as e:
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
    except Exception as e:
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
    except Exception as e:
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


def format_response(state: VisionState) -> Dict[str, Any]:
    """Formats the vision pipeline results into an AI message."""
    error = state.get("error")
    if error:
        return {"messages": [AIMessage(content=f"Pipeline stopped: {error}")]}
    
    final = state.get("final", [])
    if not final:
        return {"messages": [AIMessage(content="No objects found matching the prompt.")]}
        
    response = f"Found {len(final)} objects in `{state.get('image_path')}`:\n"
    for i, obj in enumerate(final):
        response += f"- **{obj['label']}** (confidence: {obj['score']})\n"
        
    return {"messages": [AIMessage(content=response)]}