from __future__ import annotations

import os
from pathlib import Path

import requests


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return

    print(f"[download] {url} -> {dest}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"[ok] {dest}")


def main():
    base = Path("weights")

    gdino_weights = base / "gdino.pth"
    gdino_cfg = base / "gdino_config.py"
    sam2_weights = base / "sam2.pt"

    download(
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        gdino_weights,
    )
    download(
        "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        gdino_cfg,
    )
    download(
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        sam2_weights,
    )


if __name__ == "__main__":
    main()

