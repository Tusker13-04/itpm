from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from pipeline.nodes import node_clip, node_filter, node_gdino, node_sam2, process_message, format_response
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

    graph.add_node("process_message", process_message)
    graph.add_node("gdino", node_gdino)
    graph.add_node("sam2", node_sam2)
    graph.add_node("clip", node_clip)
    graph.add_node("filter", node_filter)
    graph.add_node("format_response", format_response)

    graph.add_edge(START, "process_message")
    graph.add_edge("process_message", "gdino")

    graph.add_conditional_edges(
        "gdino",
        route_after_gdino,
        {
            "continue": "sam2",
            "abort": "format_response",
        },
    )

    graph.add_edge("sam2", "clip")
    graph.add_edge("clip", "filter")
    graph.add_edge("filter", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()


pipeline = build_pipeline()