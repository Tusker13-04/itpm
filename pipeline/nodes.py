from __future__ import annotations

import logging
import base64
import os
import tempfile
import uuid
import cv2
import numpy as np
import time
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

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block["text"])
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
                                log.error("Failed to decode image from UI (type: image): %s", e)
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
                                log.error("Failed to decode image from UI (type: image_url): %s", e)
                elif isinstance(block, str):
                    text_parts.append(block)
            content_str = " ".join(text_parts).strip()
        else:
            content_str = str(content)
            
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
        start_time = time.time()
        result = run_grounding_dino.invoke(
            {
                "image_path": image_path,
                "prompt": prompt,
            }
        )
        gdino_time = time.time() - start_time

        if not result["boxes"]:
            return {"boxes": [], "phrases": [], "logits": [], "error": "no_detections", "gdino_time": gdino_time}

        return {
            "boxes": result["boxes"],
            "phrases": result["phrases"],
            "logits": result["logits"],
            "error": None,
            "gdino_time": gdino_time,
        }
    except Exception as e:
        return {"error": str(e), "boxes": []}


def node_sam2(state: VisionState) -> Dict[str, Any]:
    """Node wrapper around the SAM2 segmentation tool."""
    try:
        start_time = time.time()
        result = run_sam2.invoke(
            {
                "image_path": state["image_path"],
                "boxes": state["boxes"],
            }
        )
        sam2_time = time.time() - start_time
        return {"masks": result["masks"], "sam2_time": sam2_time}
    except Exception as e:
        return {"error": str(e), "masks": []}


def node_clip(state: VisionState) -> Dict[str, Any]:
    """Node wrapper around CLIP re-ranking."""
    try:
        start_time = time.time()
        result = run_clip_rerank.invoke(
            {
                "image_path": state["image_path"],
                "boxes": state["boxes"],
                "phrases": state["phrases"],
            }
        )
        clip_time = time.time() - start_time
        return {"clip_scores": result["clip_scores"], "clip_time": clip_time}
    except Exception as e:
        return {"error": str(e), "clip_scores": []}


def node_filter(state: VisionState) -> Dict[str, Any]:
    """Filters detections using CLIP scores and packs final results."""
    boxes = state.get("boxes") or []
    phrases = state.get("phrases") or []
    masks = state.get("masks") or []
    clip_scores = state.get("clip_scores") or []

    final = []
    for box, phrase, mask, score in zip(boxes, phrases, masks, clip_scores):
        if score >= CLIP_THRESHOLD:
            final.append({
                "label": phrase,
                "box": box,
                "mask": mask,
                "score": round(float(score), 4),
            })
    return {"final": final}


def format_response(state: VisionState) -> Dict[str, Any]:
    """Formats results, draws bounding boxes on the image, and returns a multimodal AIMessage."""
    error = state.get("error")
    if error:
        return {"messages": [AIMessage(content=f"Pipeline stopped: {error}")]}
    
    final = state.get("final", [])
    if not final:
        return {"messages": [AIMessage(content="No objects found matching the prompt.")]}
        
    image_path = state.get("image_path")
    if not image_path or not os.path.exists(image_path):
        return {"messages": [AIMessage(content="Results found, but original image path is missing for visualization.")]}

    img = cv2.imread(image_path)
    h, w, _ = img.shape

    unique_labels = list(set([obj['label'] for obj in final]))
    np.random.seed(42)
    colors = {label: [int(c) for c in np.random.randint(0, 255, 3)] for label in unique_labels}

    for obj in final:
        cx, cy, bw, bh = obj["box"]
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        
        color = colors[obj["label"]]
        thickness = max(2, int(min(h, w) * 0.005))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        text = f"{obj['label']} {obj['score']:.2f}"
        font_scale = max(0.5, min(h, w) * 0.001)
        font_thick = max(1, int(thickness / 2))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick)

    # Save image with inference time written on it
    gdino_time = state.get("gdino_time", 0)
    sam2_time = state.get("sam2_time", 0)
    clip_time = state.get("clip_time", 0)
    total_time = gdino_time + sam2_time + clip_time
    
    time_text = f"GDINO+SAM2+CLIP: {total_time:.3f}s"
    cv2.putText(img, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    output_path = os.path.join(tempfile.gettempdir(), f"gdino_result_{uuid.uuid4().hex[:8]}.jpg")
    cv2.imwrite(output_path, img)
    log.info(f"[GDINO Pipeline] Saved result to {output_path}")

    _, buffer = cv2.imencode('.jpg', img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    img_url = f"data:image/jpeg;base64,{img_b64}"

    text_response = f"**Grounding DINO + SAM2 + CLIP Pipeline**\n\n"
    text_response += f"Total inference time: **{total_time:.3f}s**\n\n"
    text_response += f"Found {len(final)} objects:\n"
    for i, obj in enumerate(final):
        text_response += f"- **{obj['label']}** (confidence: {obj['score']})\n"

    message = AIMessage(
        content=[
            {"type": "text", "text": text_response},
            {"type": "image_url", "image_url": {"url": img_url}}
        ]
    )

    return {"messages": [message]}


def node_yolo_compare(state: VisionState) -> Dict[str, Any]:
    """Runs YOLOE-26 open-vocabulary inference on the same image for comparison."""
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("[YOLOE] ultralytics not installed. Install via: pip install ultralytics")
        return {"messages": [AIMessage(content="YOLOE-26 comparison skipped (ultralytics not installed).")]}

    image_path = state.get("image_path")
    prompt = state.get("prompt")
    
    if not image_path or not os.path.exists(image_path):
        return {"messages": [AIMessage(content="YOLOE-26 comparison skipped (image path missing).")]}
    
    if not prompt:
        return {"messages": [AIMessage(content="YOLOE-26 comparison skipped (no text prompt provided).")]}

    try:
        # Load YOLOE-26 open-vocabulary segmentation model
        model = YOLO("yoloe-26l-seg.pt")
        
        # Parse prompt: convert "person . animal . car" -> ["person", "animal", "car"]
        classes = [cls.strip() for cls in prompt.split(".") if cls.strip()]
        
        # Set text prompts for YOLOE-26
        model.set_classes(classes)
        log.info(f"[YOLOE] Set text prompts: {classes}")
        
        start_time = time.time()
        results = model.predict(image_path, verbose=False)
        yoloe_time = time.time() - start_time
        
        # Draw results on image
        img = cv2.imread(image_path)
        h, w, _ = img.shape
        
        detections = results[0].boxes
        num_detections = len(detections)
        
        np.random.seed(123)
        colors_yoloe = {}
        
        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = classes[cls] if cls < len(classes) else f"class_{cls}"
            
            if label not in colors_yoloe:
                colors_yoloe[label] = [int(c) for c in np.random.randint(0, 255, 3)]
            color = colors_yoloe[label]
            
            thickness = max(2, int(min(h, w) * 0.005))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            text = f"{label} {conf:.2f}"
            font_scale = max(0.5, min(h, w) * 0.001)
            font_thick = max(1, int(thickness / 2))
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick)
        
        # Write inference time on image
        time_text = f"YOLOE-26: {yoloe_time:.3f}s"
        cv2.putText(img, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        output_path = os.path.join(tempfile.gettempdir(), f"yoloe_result_{uuid.uuid4().hex[:8]}.jpg")
        cv2.imwrite(output_path, img)
        log.info(f"[YOLOE] Saved result to {output_path}")
        
        _, buffer = cv2.imencode('.jpg', img)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        img_url = f"data:image/jpeg;base64,{img_b64}"
        
        text_response = f"\n\n**YOLOE-26 Open-Vocabulary Comparison**\n\n"
        text_response += f"Total inference time: **{yoloe_time:.3f}s**\n\n"
        text_response += f"Detected {num_detections} objects using text prompts: {', '.join(classes)}.\n"
        
        message = AIMessage(
            content=[
                {"type": "text", "text": text_response},
                {"type": "image_url", "image_url": {"url": img_url}}
            ]
        )
        
        return {"messages": [message]}
        
    except Exception as e:
        log.error(f"[YOLOE] Error during inference: {e}")
        return {"messages": [AIMessage(content=f"YOLOE-26 comparison failed: {str(e)}")]}