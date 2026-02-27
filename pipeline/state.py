from __future__ import annotations

from typing import List, Any
from typing_extensions import TypedDict
from langgraph.graph import MessagesState


class VisionState(MessagesState):
    """State that extends MessagesState to enable Chat mode in LangGraph Studio."""
    image_path: str
    prompt: str
    boxes: List[List[float]]
    phrases: List[str]
    logits: List[float]
    masks: List[Any]
    clip_scores: List[float]
    final: List[dict]
    error: str
    gdino_time: float
    sam2_time: float
    clip_time: float
    gdino_result_msg: str
    gdino_result_img: str
    yolo_result_msg: str
    yolo_result_img: str