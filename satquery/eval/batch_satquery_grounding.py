"""
satquery/eval/batch_satquery_grounding.py

Batch text-guided region-grounding evaluation, mirroring GeoChat's
`geochat/eval/batch_geochat_grounding.py`. Intended for VRSBench's
visual-grounding split.

Usage:
    python -m satquery.eval.batch_satquery_grounding \
        --questions-file vrsbench_grounding.json \
        --answers-file grounding_predictions.jsonl
"""
from __future__ import annotations

import argparse
import json

from satquery.tools.grounding import run_grounding
from satquery.utils.image_io import load_image


def main():
    parser = argparse.ArgumentParser(description="Batch grounding evaluation (VRSBench)")
    parser.add_argument("--questions-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    args = parser.parse_args()

    with open(args.questions_file) as f:
        items = json.load(f)

    with open(args.answers_file, "w") as out:
        for item in items:
            arr, _ = load_image(item["image"])
            result = run_grounding(arr, item["referring_expression"])
            out.write(json.dumps({
                "question_id": item.get("question_id"),
                "referring_expression": item["referring_expression"],
                "bbox_normalized": result["bbox_normalized"],
                "confidence": result["confidence"],
            }) + "\n")

    print(f"Wrote {len(items)} predictions to {args.answers_file}")


if __name__ == "__main__":
    main()
