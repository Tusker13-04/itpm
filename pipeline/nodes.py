from __future__ import annotations

import logging
import base64
import os
import tempfile
import uuid
import json
from typing import Dict, Any

from langchain_core.messages import AIMessage, HumanMessage

from pipeline.state import VisionState
from pipeline.tools import run_grounding_dino, run_sam2, run_clip_rerank

log = logging.getLogger(__name__)

CLIP_THRESHOLD = 0.20


def process_message(state: VisionState) -> Dict[str, Any]:
    """Extracts prompt and image from the last user message."""
    messages = state.get("messages", [])
    if not messages:
        return {}
    
    last_message = messages[-1]
    result_state = {}
    
    if isinstance(last_message, HumanMessage):
        content = last_message.content
        
        # Log the exact payload structure to see what LangGraph Studio is actually sending
        try:
            log.info("[CHAT DEBUG] Message content type: %s", type(content))
            log.info("[CHAT DEBUG] Message content: %s", str(content)[:500] + "..." if len(str(content)) > 500 else str(content))
        except Exception:
            pass

        # Multimodal inputs from LangGraph Studio UI are usually represented as lists
        if isinstance(content, list):
            text_parts = []
            for block in content:
                # If block is a dict, it might have type 'text' or 'image_url'
                if isinstance(block, dict):
                    # Text blocks
                    if block.get("type") == "text":
                        text_parts.append(block["text"])
                    
                    # Image URL blocks (Base64)
                    elif block.get("type") == "image_url":
                        # LangChain standard format: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
                        # Sometimes UI sends: {"type": "image_url", "url": "data:..."} or similar variations
                        
                        image_data = block.get("image_url", {})
                        if isinstance(image_data, str):
                            image_url = image_data
                        elif isinstance(image_data, dict):
                            image_url = image_data.get("url", "")
                        else:
                            image_url = block.get("url", "")

                        if image_url.startswith("data:image"):
                            try:
                                # Parse out the base64 string
                                header, b64_data = image_url.split(",", 1)
                                
                                # Create a temporary file with a unique name
                                temp_dir = tempfile.gettempdir()
                                file_ext = ".png" if "png" in header.lower() else ".jpg"
                                temp_path = os.path.join(temp_dir, f"lg_upload_{uuid.uuid4().hex[:8]}{file_ext}")
                                
                                with open(temp_path, "wb") as f:
                                    f.write(base64.b64decode(b64_data))
                                    
                                result_state["image_path"] = temp_path
                                log.info("[CHAT] Saved uploaded image to %s", temp_path)
                            except Exception as e:
                                log.error("Failed to decode image from UI: %s", e)
                        else:
                            log.warning("Image URL did not start with 'data:image', was: %s", image_url[:50])
                            
                # Fallback: if block is just a string
                elif isinstance(block, str):
                    text_parts.append(block)
                    
            content_str = " ".join(text_parts).strip()
        else:
            content_str = str(content)
            
        log.info("[CHAT] Extracted string message: '%s'", content_str)
        
        # Fallback manual path parsing if user typed [path] instead of uploading
        if content_str.startswith("[") and "]" in content_str:
            path_end = content_str.find("]")
            img_path = content_str[1:path_end].strip()
            prompt_text = content_str[path_end+1:].strip()
            result_state["image_path"] = img_path
            result_state["prompt"] = prompt_text
        elif content_str:
            result_state["prompt"] = content_str
            
    return result_state


def node_gdino(state: VisionState) -> Dict[str, Any]:
    """Node wrapper around the Grounding DINO tool."""
    image_path = state.get("image_path")
    prompt = state.get("prompt")
    
    log.info("[GDINO] image_path='%s', prompt='%s'", image_path, prompt)
    
    if not image_path:
        return {"error": "Please provide an image. You can click the '+' icon in the chat to upload one. (Check console logs for debug info)", "boxes": []}
    if not prompt:
        return {"error": "Please provide a text prompt to search for.", "boxes": []}

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
        
    response = f"Found {len(final)} objects:\n"
    for i, obj in enumerate(final):
        response += f"- **{obj['label']}** (confidence: {obj['score']})\n"
        
    return {"messages": [AIMessage(content=response)]}