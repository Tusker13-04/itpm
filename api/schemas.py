from typing import List, Optional

from pydantic import BaseModel


class DetectRequest(BaseModel):
    image_path: str
    prompt: str
    box_thresh: float = 0.35
    clip_thresh: float = 0.20


class Detection(BaseModel):
    label: str
    box: List[float]
    score: float


class DetectResponse(BaseModel):
    detections: List[Detection]
    total: int
    error: Optional[str] = None

from typing import List, Optional

from pydantic import BaseModel


class DetectRequest(BaseModel):
    image_path: str
    prompt: str
    box_thresh: float = 0.35
    clip_thresh: float = 0.20


class Detection(BaseModel):
    label: str
    box: List[float]
    score: float


class DetectResponse(BaseModel):
    detections: List[Detection]
    total: int
    error: Optional[str] = None

