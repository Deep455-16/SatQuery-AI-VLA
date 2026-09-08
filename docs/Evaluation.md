# SatQuery AI — Evaluation Framework

## Overview

This document describes the evaluation methodology for SatQuery AI.
**No metrics are fabricated.** If a benchmark has not been run, it is documented as "Not evaluated."

---

## Task Coverage

| Task | Benchmark | Status |
|------|-----------|--------|
| Single-image VQA | RSVQA (LR/HR splits) | Not evaluated |
| Image captioning | VRSBench captioning split | Not evaluated |
| Visual grounding | VRSBench grounding split | Not evaluated (no specialist grounding model) |
| Change detection | LEVIR-CD, WHU-CD | Not evaluated |
| Optical+SAR analysis | BigEarthNet fusion split | Not evaluated |

---

## Metrics Defined

### VQA (RSVQA)
- **Accuracy:** Exact match against ground-truth answers
- **Semantic match:** Token overlap (ROUGE-1) for open-ended answers

### Captioning (VRSBench)
- **CIDEr:** Consensus-based image description evaluation
- **BLEU-4:** n-gram precision
- **ROUGE-L:** Longest common subsequence recall

### Grounding (VRSBench)
- **IoU (Intersection over Union):** Overlap between predicted and ground-truth bounding boxes
- **Acc@0.5:** Fraction of predictions with IoU ≥ 0.5
- **Note:** Not applicable in current deployment (no specialist grounding model)

### Change Detection (LEVIR-CD)
- **Precision:** Changed pixels correctly identified / all predicted changed pixels
- **Recall:** Changed pixels correctly identified / all actual changed pixels
- **F1 Score:** Harmonic mean of precision and recall
- **IoU:** Overlap of predicted and ground-truth change masks
- **Note:** OpenCV-based change detection is not trained/calibrated on LEVIR-CD

---

## Current Evaluation Status

### Why No Metrics Are Reported Yet

1. **Cloud reasoning layer (Gemini):** Cannot be fine-tuned or evaluated with
   standard VQA metrics without a large-scale benchmark run.

2. **OpenCV change detection:** Works deterministically but is not calibrated
   against labeled change-detection datasets in this deployment.

3. **No specialist models loaded:** TerraQ-VL, GeoChat, VisTA, DS_UNet are not
   integrated in the default free-cloud mode.

This is an honest limitation. Reporting fabricated metrics is worse than
reporting "not evaluated."

---

## How to Evaluate (Future Work)

```bash
# VQA evaluation
python -m satquery.eval.batch_satquery_vqa \
    --questions-file rsvqa_test_questions.json \
    --answers-file rsvqa_predictions.jsonl

# Grounding evaluation (requires specialist model)
python -m satquery.eval.batch_satquery_grounding \
    --questions-file vrsbench_grounding.json \
    --answers-file grounding_predictions.jsonl

# Change detection evaluation
python -m satquery.eval.batch_satquery_change \
    --questions-file cdvqa_test.json \
    --answers-file change_predictions.jsonl
```

---

## Anti-Hallucination Quality Assessment

While formal benchmark scores are not yet available, the system's
anti-hallucination design can be qualitatively assessed:

| Criterion | Implementation |
|-----------|---------------|
| No fake confidence scores | ✅ Evidence Strength derived from computation |
| No fake bounding boxes | ✅ Grounding returns `available=False` when no model loaded |
| No fake change masks | ✅ Change maps from OpenCV pixel comparison |
| Observation vs. inference separated | ✅ Structured output parsing |
| Uncertainty explicitly stated | ✅ Limitations field in every response |
| API key missing → honest error | ✅ Not hidden as fake answer |
