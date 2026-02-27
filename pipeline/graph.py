from __future__ import annotations

from langgraph.graph import END, StateGraph

from pipeline.nodes import node_clip, node_filter, node_gdino, node_sam2
from pipeline.state import VisionState


def route_after_gdino(state: VisionState) -> str:
    """
    Conditional router after the GDINO node.

    If there is an error or no boxes, we abort early; otherwise we continue.
    """
    if state.get("error"):
        return "abort"
    if not state.get("boxes"):
        return "abort"
    return "continue"


def build_pipeline():
    """
    Build and compile the LangGraph StateGraph for the pipeline.
    """
    graph = StateGraph(VisionState)

    graph.add_node("gdino", node_gdino)
    graph.add_node("sam2", node_sam2)
    graph.add_node("clip", node_clip)
    graph.add_node("filter", node_filter)

    graph.set_entry_point("gdino")

    graph.add_conditional_edges(
        "gdino",
        route_after_gdino,
        {
            "continue": "sam2",
            "abort": END,
        },
    )

    graph.add_edge("sam2", "clip")
    graph.add_edge("clip", "filter")
    graph.add_edge("filter", END)

    return graph.compile()


pipeline = build_pipeline()

