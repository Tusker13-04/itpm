from __future__ import annotations

from typing import Any, Dict, List, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class VisionState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    image_path: Optional[str]
    prompt: Optional[str]
    boxes: Optional[List[List[float]]]
    phrases: Optional[List[str]]
    logits: Optional[List[float]]
    masks: Optional[List[Any]]
    clip_scores: Optional[List[float]]
    final: Optional[List[Dict[str, Any]]]
    error: Optional[str]
