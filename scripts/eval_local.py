#!/usr/bin/env python3
"""Batch-evaluate bundled samples using sibling VTT (no Whisper)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import analyze_recording, find_sibling_vtt  # noqa: E402

DATASET = ROOT / "class recordings -20261022T171511Z-1-001" / "class recordings "


def main() -> None:
    if not DATASET.is_dir():
        print("Dataset not found:", DATASET)
        sys.exit(1)
    rows = []
    for mp4 in sorted(DATASET.glob("*/*.mp4")):
        label = mp4.parent.name.strip()
        vtt = find_sibling_vtt(mp4)
        if not vtt:
            print(f"{label:16} -> SKIP (no VTT)  {mp4.name}")
            continue
        result = analyze_recording(mp4, vtt_path=vtt, use_whisper=False)
        pred = result["assessment"].get("quality")
        rows.append((label, pred, mp4.name))
        print(f"{label:16} -> {pred:10}  {mp4.name}")
    print("\nDone.", len(rows), "files")


if __name__ == "__main__":
    main()
