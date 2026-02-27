from typing import List, Optional, TypedDict


class VisionState(TypedDict, total=False):
    """
    Shared mutable state passed between LangGraph nodes.

    This mirrors the spec in the project document.
    """

    image_path: str
    prompt: str  # "person . laptop . backpack"

    # Outputs from Grounding DINO
    boxes: Optional[List]  # [[x1,y1,x2,y2], ...] normalized
    phrases: Optional[List]  # ["person", "laptop"]
    logits: Optional[List]  # GDINO confidence scores

    # Outputs from SAM2
    masks: Optional[List]  # binary masks from SAM2

    # Outputs from CLIP re-ranking
    clip_scores: Optional[List]  # CLIP re-rank scores per box

    # Final filtered detections
    final: Optional[List]  # [{"label","box","mask","score"}, ...]

    # Error/debugging info
    error: Optional[str]  # debug branch signal

