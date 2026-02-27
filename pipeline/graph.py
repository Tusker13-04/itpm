from langgraph.graph import StateGraph, END
from pipeline.state import VisionState
from pipeline.nodes import (
    process_message,
    node_gdino,
    node_sam2,
    node_clip,
    node_filter,
    format_response,
    node_yolo_compare,
)


def should_continue_to_sam(state):
    """Router: if boxes are empty or error, skip to YOLO."""
    error = state.get("error")
    boxes = state.get("boxes", [])
    if error or not boxes:
        return "yolo_compare"
    return "sam2"


def build_graph():
    workflow = StateGraph(VisionState)

    # Add all nodes
    workflow.add_node("process_message", process_message)
    workflow.add_node("gdino", node_gdino)
    workflow.add_node("sam2", node_sam2)
    workflow.add_node("clip", node_clip)
    workflow.add_node("filter", node_filter)
    workflow.add_node("format_response", format_response)
    workflow.add_node("yolo_compare", node_yolo_compare)

    # Set entry point
    workflow.set_entry_point("process_message")
    
    # GDINO Pipeline
    workflow.add_edge("process_message", "gdino")
    workflow.add_conditional_edges(
        "gdino",
        should_continue_to_sam,
        {
            "sam2": "sam2",
            "yolo_compare": "yolo_compare"
        }
    )
    workflow.add_edge("sam2", "clip")
    workflow.add_edge("clip", "filter")
    workflow.add_edge("filter", "format_response")
    workflow.add_edge("format_response", "yolo_compare")
    
    # YOLO runs after GDINO (sequential but independent)
    workflow.add_edge("yolo_compare", END)

    return workflow.compile()


graph = build_graph()