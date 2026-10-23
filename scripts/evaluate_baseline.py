
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from src.pipeline import analyze_recording, find_sibling_vtt

# The four required output classes.
# Confirm that "weak" is the dataset's equivalent of "Poor".
LABEL_MAP = {
    "weak": "Poor",
    "average": "Average",
    "good": "Good",
    "excellent": "Excellent",
}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("baseline_results")
    )
    args = parser.parse_args()

    if not args.data.is_dir():
        raise SystemExit(f"Dataset directory not found: {args.data}")

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []

    for folder, true_label in LABEL_MAP.items():
        class_dir = args.data / folder

        if not class_dir.is_dir():
            print(f"[SKIP] Missing folder: {class_dir}")
            continue

        videos = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )

        for video in videos:
            print(f"\n[{true_label}] {video.name}")

            row = {
                "video": str(video),
                "true_label": true_label,
                "predicted_label": "",
                "confidence": "",
                "engine": "",
                "model": "",
                "error": "",
            }

            try:
                result = analyze_recording(
                    video,
                    vtt_path=find_sibling_vtt(video),
                    use_whisper=True,
                    on_progress=lambda msg: print(f"  {msg}"),
                )

                assessment = result["assessment"]
                row["predicted_label"] = assessment.get("quality", "")
                row["confidence"] = assessment.get("confidence", "")
                row["engine"] = assessment.get("engine", "")
                row["model"] = assessment.get("model", "")

                safe_name = f"{len(rows):03d}_{video.stem}"
                (args.output / f"{safe_name}.json").write_text(
                    json.dumps(result, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            except Exception as exc:
                row["error"] = str(exc)
                print(f"  ERROR: {exc}")

            rows.append(row)
            print(
                f"  True: {row['true_label']} | "
                f"Predicted: {row['predicted_label'] or 'FAILED'}"
            )

    csv_path = args.output / "predictions.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [
            "video", "true_label", "predicted_label",
            "confidence", "engine", "model", "error"
        ])
        writer.writeheader()
        writer.writerows(rows)

    successful = [
        r for r in rows
        if r["predicted_label"] in {
            "Poor", "Average", "Good", "Excellent"
        }
    ]

    correct = sum(
        r["true_label"] == r["predicted_label"]
        for r in successful
    )

    print("\n--- BASELINE SUMMARY ---")
    print(f"Videos discovered: {len(rows)}")
    print(f"Successful predictions: {len(successful)}")
    print(f"Failed predictions: {len(rows) - len(successful)}")

    if successful:
        print(f"Accuracy: {correct}/{len(successful)} "
              f"= {correct / len(successful):.1%}")

        print("\nPredicted class distribution:")
        for label, count in Counter(
            r["predicted_label"] for r in successful
        ).items():
            print(f"  {label}: {count}")

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
