"""
satquery/eval/batch_satquery_vqa.py

Batch single-image VQA evaluation, mirroring GeoChat's
`geochat/eval/batch_geochat_vqa.py`. Intended for the RSVQA and VRSBench
VQA splits named in the problem statement.

Expected input JSON format (one entry per question), same convention as
GeoChat's evaluation jsonl/json files:

    [
      {"image": "path/to/image.png", "question": "...", "question_id": "..."},
      ...
    ]

Usage:
    python -m satquery.eval.batch_satquery_vqa \
        --questions-file rsvqa_test_questions.json \
        --answers-file rsvqa_predictions.jsonl
"""
from __future__ import annotations

import argparse
import json

from satquery.tools.vqa_caption import run_vqa
from satquery.utils.image_io import load_image


def main():
    parser = argparse.ArgumentParser(description="Batch VQA evaluation (RSVQA / VRSBench)")
    parser.add_argument("--questions-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    args = parser.parse_args()

    with open(args.questions_file) as f:
        questions = json.load(f)

    with open(args.answers_file, "w") as out:
        for q in questions:
            arr, _ = load_image(q["image"])
            result = run_vqa(arr, q["question"])
            out.write(json.dumps({
                "question_id": q.get("question_id"),
                "question": q["question"],
                "answer": result["answer"],
                "confidence": result["confidence"],
            }) + "\n")

    print(f"Wrote {len(questions)} predictions to {args.answers_file}")


if __name__ == "__main__":
    main()
