from __future__ import annotations

from unittest.mock import patch

from pipeline.nodes import node_filter, node_gdino


def base_state(**kw):
    state = {
        "image_path": "x.jpg",
        "prompt": "person",
        "boxes": None,
        "phrases": None,
        "logits": None,
        "masks": None,
        "clip_scores": None,
        "final": None,
        "error": None,
    }
    state.update(kw)
    return state


@patch("pipeline.nodes.run_grounding_dino")
def test_gdino_empty(mock_tool):
    mock_tool.invoke.return_value = {"boxes": [], "phrases": [], "logits": []}
    result = node_gdino(base_state())
    assert result["error"] == "no_detections"
    assert result["boxes"] == []


@patch("pipeline.nodes.run_grounding_dino")
def test_gdino_detects(mock_tool):
    mock_tool.invoke.return_value = {
        "boxes": [[0.1, 0.1, 0.5, 0.5]],
        "phrases": ["person"],
        "logits": [0.9],
    }
    result = node_gdino(base_state())
    assert result["error"] is None
    assert len(result["boxes"]) == 1
    assert result["phrases"][0] == "person"


def test_filter_clips_low_scores():
    state = base_state(
        boxes=[[0, 0, 1, 1], [0, 0, 1, 1]],
        phrases=["person", "laptop"],
        masks=[[[1]], [[1]]],
        clip_scores=[0.8, 0.05],
    )
    result = node_filter(state)
    assert len(result["final"]) == 1
    assert result["final"][0]["label"] == "person"

