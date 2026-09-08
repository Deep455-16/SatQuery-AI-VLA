"""
satquery/eval/batch_satquery_change.py

Batch bi-temporal change-VQA / change-description evaluation. This task
has no GeoChat equivalent (GeoChat is single-image only) — it targets
the CDVQA benchmark named in the problem statement.

Expected input JSON:
    [
      {"image_a": "t1.png", "image_b": "t2.png", "question": "...", "question_id": "..."},
      ...
    ]

Usage:
    python -m satquery.eval.batch_satquery_change \
        --questions-file cdvqa_test.json \
        --answers-file change_predictions.jsonl
"""
from __future__ import annotations

import argparse
import json

from satquery.tools.change import run_change_analysis
from satquery.utils.image_io import load_image


def main():
    parser = argparse.ArgumentParser(description="Batch change-VQA evaluation (CDVQA)")
    parser.add_argument("--questions-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    args = parser.parse_args()

    with open(args.questions_file) as f:
        items = json.load(f)

    with open(args.answers_file, "w") as out:
        for item in items:
            arr_a, _ = load_image(item["image_a"])
            arr_b, _ = load_image(item["image_b"])
            result = run_change_analysis(arr_a, arr_b, item.get("question"))
            out.write(json.dumps({
                "question_id": item.get("question_id"),
                "question": item.get("question"),
                "answer": result["answer"],
                "confidence": result["confidence"],
            }) + "\n")

    print(f"Wrote {len(items)} predictions to {args.answers_file}")


if __name__ == "__main__":
    main()
