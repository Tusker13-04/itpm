from __future__ import annotations

from typing import Annotated, TypedDict, List, Any
from langgraph.graph.message import add_messages


class VisionState(TypedDict):
    messages: Annotated[list, add_messages]
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