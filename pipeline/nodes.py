from __future__ import annotations

import logging
import base64
import os
import tempfile
import uuid
import json
import random
from typing import Dict, Any

import cv2
import numpy as np

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

        # Multimodal inputs from LangGraph Studio UI are represented as lists
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    # Text blocks
                    if block.get("type") == "text":
                        text_parts.append(block["text"])
                    
                    # LangGraph Studio UI raw image block format
                    elif block.get("type") == "image":
                        b64_data = block.get("data", "")
                        if b64_data:
                            try:
                                temp_dir = tempfile.gettempdir()
                                temp_path = os.path.join(temp_dir, f"lg_upload_{uuid.uuid4().hex[:8]}.jpg")
                                with open(temp_path, "wb") as f:
                                    f.write(base64.b64decode(b64_data))
                                result_state["image_path"] = temp_path
                            except Exception as e:
                                log.error("Failed to decode image from UI: %s", e)

                    # Standard LangChain image URL block
                    elif block.get("type") == "image_url":
                        image_data = block.get("image_url", {})
                        image_url = image_data if isinstance(image_data, str) else image_data.get("url", "")
                        if image_url.startswith("data:image"):
                            try:
                                header, b64_data = image_url.split(",", 1)
                                temp_dir = tempfile.gettempdir()
                                file_ext = ".png" if "png" in header.lower() else ".jpg"
                                temp_path = os.path.join(temp_dir, f"lg_upload_{uuid.uuid4().hex[:8]}{file_ext}")
                                with open(temp_path, "wb") as f:
                                    f.write(base64.b64decode(b64_data))
                                result_state["image_path"] = temp_path
                            except Exception as e:
                                log.error("Failed to decode image_url from UI: %s", e)
                                
                elif isinstance(block, str):
                    text_parts.append(block)
                    
            content_str = " ".join(text_parts).strip()
        else:
            content_str = str(content)
            
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
    
    if not image_path:
        return {"error": "Please provide an image. You can click the '+' icon in the chat to upload one.", "boxes": []}
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
            return {"boxes": [], "phrases": [], "logits": [], "error": "no_detections"}

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
    try:
        result = run_sam2.invoke(
            {
                "image_path": state["image_path"],
                "boxes": state["boxes"],
            }
        )
        return {"masks": result["masks"]}
    except Exception as e:
        log.error("[SAM2] %s", e)
        return {"error": str(e), "masks": []}


def node_clip(state: VisionState) -> Dict[str, Any]:
    """Node wrapper around CLIP re-ranking."""
    try:
        result = run_clip_rerank.invoke(
            {
                "image_path": state["image_path"],
                "boxes": state["boxes"],
                "phrases": state["phrases"],
            }
        )
        return {"clip_scores": result["clip_scores"]}
    except Exception as e:
        log.error("[CLIP] %s", e)
        return {"error": str(e), "clip_scores": []}


def node_filter(state: VisionState) -> Dict[str, Any]:
    """
    Filters detections using CLIP scores and packs final results.
    """
    boxes = state.get("boxes") or []
    phrases = state.get("phrases") or []
    masks = state.get("masks") or []
    clip_scores = state.get("clip_scores") or []

    final = [
        {
            "label": phrase,
            "box": box, # Note: These are [cx, cy, w, h] normalized
            "mask": mask,
            "score": round(float(score), 4),
        }
        for box, phrase, mask, score in zip(boxes, phrases, masks, clip_scores)
        if score >= CLIP_THRESHOLD
    ]

    return {"final": final}


def draw_results_on_image(image_path: str, final_detections: List[Dict[str, Any]]) -> str:
    """
    Draws bounding boxes and labels on the image using distinct random colors per class.
    Returns base64 encoded string of the annotated image.
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Generate a unique random color for each distinct class label
    unique_labels = list(set([d["label"] for d in final_detections]))
    color_map = {}
    for label in unique_labels:
        # BGR format for OpenCV
        color_map[label] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

    for det in final_detections:
        label = det["label"]
        score = det["score"]
        box = det["box"] # [cx, cy, norm_w, norm_h]
        color = color_map[label]

        # Convert normalized center coords to absolute pixel corners
        cx, cy, bw, bh = box[0], box[1], box[2], box[3]
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # Draw text background and text
        text = f"{label} {score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Encode to base64 to send back to LangGraph UI
    _, buffer = cv2.imencode('.jpg', img)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"


def format_response(state: VisionState) -> Dict[str, Any]:
    """Formats the vision pipeline results into an AI message, including the annotated image."""
    error = state.get("error")
    if error:
        return {"messages": [AIMessage(content=f"Pipeline stopped: {error}")]}
    
    final = state.get("final", [])
    if not final:
        return {"messages": [AIMessage(content="No objects found matching the prompt.")]}
        
    text_response = f"Found {len(final)} objects:\n"
    for i, obj in enumerate(final):
        text_response += f"- **{obj['label']}** (confidence: {obj['score']})\n"
        
    # Generate the annotated image
    try:
        image_path = state.get("image_path")
        b64_image = draw_results_on_image(image_path, final)
        
        # LangGraph UI can render multimodal AI messages
        ai_message = AIMessage(
            content=[
                {"type": "text", "text": text_response},
                {"type": "image_url", "image_url": {"url": b64_image}}
            ]
        )
        return {"messages": [ai_message]}
    except Exception as e:
        log.error("Failed to draw bounding boxes: %s", e)
        return {"messages": [AIMessage(content=text_response + f"\n\n*(Failed to generate output image: {e})*")]}