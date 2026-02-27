from __future__ import annotations

from typing import Annotated, List, Any, Optional
from typing_extensions import TypedDict, NotRequired
from langgraph.graph.message import add_messages


class VisionState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    image_path: NotRequired[str]
    prompt: NotRequired[str]
    boxes: NotRequired[List[List[float]]]
    phrases: NotRequired[List[str]]
    logits: NotRequired[List[float]]
    masks: NotRequired[List[Any]]
    clip_scores: NotRequired[List[float]]
    final: NotRequired[List[dict]]
    error: NotRequired[str]
    gdino_time: NotRequired[float]
    sam2_time: NotRequired[float]
    clip_time: NotRequired[float]
    gdino_result_msg: NotRequired[str]
    gdino_result_img: NotRequired[str]
    yolo_result_msg: NotRequired[str]
    yolo_result_img: NotRequired[str]