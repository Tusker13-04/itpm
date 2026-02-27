from __future__ import annotations

from pipeline.graph import pipeline


def print_ascii() -> None:
    """Print an ASCII diagram of the pipeline graph."""
    print(pipeline.get_graph().draw_ascii())


def save_mermaid_png(path: str = "graph.png") -> None:
    """
    Render a diagram PNG via Mermaid's API.

    Requires internet connectivity as it uses mermaid.ink under the hood.
    """
    from langchain_core.runnables.graph import MermaidDrawMethod

    png = pipeline.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API
    )
    with open(path, "wb") as f:
        f.write(png)
    print(f"Saved → {path}")


def print_mermaid_code() -> None:
    """Print Mermaid code suitable for pasting into mermaid.live."""
    print(pipeline.get_graph().draw_mermaid())


if __name__ == "__main__":
    print_ascii()
    print_mermaid_code()

from __future__ import annotations

from pipeline.graph import pipeline


def print_ascii() -> None:
    """Print an ASCII diagram of the LangGraph pipeline."""
    print(pipeline.get_graph().draw_ascii())


def save_mermaid_png(path: str = "graph.png") -> None:
    """
    Render the pipeline as a Mermaid PNG.

    Requires internet access, as this uses mermaid.ink via
    `MermaidDrawMethod.API`.
    """
    from langchain_core.runnables.graph import MermaidDrawMethod

    png = pipeline.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
    )
    with open(path, "wb") as f:
        f.write(png)
    print(f"Saved → {path}")


def print_mermaid_code() -> None:
    """Print raw Mermaid diagram code (for mermaid.live etc.)."""
    print(pipeline.get_graph().draw_mermaid())


if __name__ == "__main__":
    print_ascii()
    print_mermaid_code()

