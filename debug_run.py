from __future__ import annotations

import logging

from pipeline.graph import pipeline

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    """
    Run a sample debug pass through the pipeline.

    Expects a `test.jpg` image in the working directory or a valid path
    if you edit the `image_path` below.
    """
    initial_state = {
        "image_path": "test.jpg",
        "prompt": "person . laptop",
        "boxes": None,
        "phrases": None,
        "logits": None,
        "masks": None,
        "clip_scores": None,
        "final": None,
        "error": None,
    }

    # Stream: see each node's state delta live
    for step in pipeline.stream(initial_state):
        node = list(step.keys())[0]
        print("\n" + "=" * 40)
        print(f"NODE: {node}")
        print(f"DELTA: {step[node]}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import logging

from pipeline.graph import pipeline


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    # NOTE: Replace 'test.jpg' with an actual image path on your system.
    initial_state = {
        "image_path": "test.jpg",
        "prompt": "person . laptop",
        "boxes": None,
        "phrases": None,
        "logits": None,
        "masks": None,
        "clip_scores": None,
        "final": None,
        "error": None,
    }

    # Stream mode: see each node's state delta live.
    for step in pipeline.stream(initial_state):
        node = list(step.keys())[0]
        print("\n" + "=" * 40)
        print(f"NODE: {node}")
        print(f"DELTA: {step[node]}")


if __name__ == "__main__":
    main()

