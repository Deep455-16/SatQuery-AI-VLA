# SatQuery AI — System Architecture

## Overview

SatQuery AI is an **agentic, evidence-first** satellite image analysis system.
It deliberately separates three concerns:

1. **Deterministic preprocessing** — what computers can compute exactly
2. **Evidence assembly** — structured, verifiable measurements
3. **Multimodal reasoning** — what an LLM does best

This separation prevents the common failure mode of asking a generic LLM to
"perform scientific remote sensing" on raw pixels.

---

## Full System Architecture

```
                              USER
                               │
                  ┌────────────▼────────────┐
                  │    STREAMLIT FRONTEND    │
                  │  (satquery/serve/)       │
                  │                         │
                  │  • Upload interface      │
                  │  • Layer visualization   │
                  │  • Structured results    │
                  │  • Analysis history      │
                  └────────────┬────────────┘
                               │ query + image bytes
                  ┌────────────▼────────────┐
                  │   INPUT VALIDATION       │
                  │ (controller/input_validator.py) │
                  │                         │
                  │  Classifies:            │
                  │  • single image          │
                  │  • optical + SAR pair    │
                  │  • bi-temporal pair      │
                  │  Validates:             │
                  │  • dimensions, CRS      │
                  └────────────┬────────────┘
                               │ classified input
                  ┌────────────▼────────────┐
                  │   SATELLITE PREPROCESSING│
                  │  (utils/image_preprocessor.py) │
                  │                         │
                  │  • GeoTIFF parsing       │
                  │  • Band normalization     │
                  │  • RGB/false-color viz    │
                  │  • SAR log-normalization  │
                  │  • NDVI computation       │
                  │  • Image statistics       │
                  │  • Resize for LLM         │
                  └────────────┬────────────┘
                               │ PIL images + stats
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   SINGLE IMAGE          OPTICAL + SAR        BEFORE + AFTER
   ─────────────          ─────────────        ─────────────
   • RGB viz              • Dual viz            • RGB viz ×2
   • False color          • SAR enhancement     • OpenCV change
   • NDVI                 • Modality notes       detection
   • Band stats           • Band stats ×2       • Change mask
                                                 • Changed area %
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                  ┌────────────▼────────────┐
                  │    EVIDENCE LAYER        │
                  │ (controller/evidence_layer.py) │
                  │                         │
                  │  Assembles:             │
                  │  • Image metadata       │
                  │  • Computed statistics  │
                  │  • Change evidence      │
                  │  • Modality context     │
                  │  • Preprocessing notes  │
                  └────────────┬────────────┘
                               │ evidence context string
                  ┌────────────▼────────────┐
                  │  GEMINI 2.5 FLASH API   │
                  │  (adapters/gemini_adapter.py) │
                  │                         │
                  │  System prompt:         │
                  │  "Answer only from      │
                  │   supplied image and    │
                  │   verified evidence..."  │
                  │                         │
                  │  Input:                 │
                  │  • image visualization  │
                  │  • evidence context     │
                  │  • user query           │
                  │                         │
                  │  Output (structured):   │
                  │  • ANSWER               │
                  │  • OBSERVATIONS         │
                  │  • INFERENCES           │
                  │  • EVIDENCE_STRENGTH    │
                  │  • LIMITATIONS          │
                  └────────────┬────────────┘
                               │
                  ┌────────────▼────────────┐
                  │   MONGODB ATLAS M0       │
                  │  (database/mongodb.py)  │
                  │                         │
                  │  Stores:                │
                  │  • Session metadata     │
                  │  • Image metadata       │
                  │  • Analysis results     │
                  │  • Model run logs       │
                  │  • User feedback        │
                  │                         │
                  │  Failure: non-fatal,    │
                  │  analysis still shown   │
                  └────────────┬────────────┘
                               │
                              USER
                  (structured answer + change map
                   + layer visualization + PDF)
```

---

## Why Gemini Is NOT Treated as a Scientific RS Model

General-purpose vision-language models, including Gemini, have the following
limitations in satellite image analysis:

1. **No calibrated spectral knowledge** — they cannot reliably compute NDVI,
   backscatter statistics, or other quantitative remote-sensing measurements.

2. **Hallucination risk** — without grounding in actual pixel values, LLMs tend
   to "complete" plausible-sounding descriptions that may be factually wrong.

3. **No temporal reasoning** — a single LLM call with two images cannot perform
   actual pixel-level change detection.

Therefore SatQuery AI uses Gemini **only for what it excels at**:
- Natural language understanding (parsing the user's question)
- Multimodal pattern recognition (identifying visual patterns in a quality image)
- Narrative synthesis (combining evidence into a coherent explanation)
- Uncertainty communication (expressing what cannot be determined)

All quantitative measurements (NDVI, change percentage, band statistics)
are computed deterministically by local algorithms and **provided to Gemini
as evidence context**, not left for it to guess.

---

## Anti-Hallucination Design

1. **System prompt constraint:** Gemini is instructed not to invent facts.
2. **Evidence grounding:** Computed values are provided in the prompt.
3. **Structured output parsing:** Observations vs. Inferences vs. Limitations
   are parsed separately and displayed distinctly in the UI.
4. **No fake confidence:** Evidence Strength labels (High/Moderate/Limited)
   are derived from computed change metrics, not random numbers.
5. **No fake bounding boxes:** Grounding is disabled (returns honest limitation)
   unless a real specialist model is loaded.
6. **No fake change masks:** Change maps are produced by OpenCV pixel comparison,
   not hallucinated by the LLM.

---

## Module Map

```
satquery/
├── adapters/
│   └── gemini_adapter.py      # Gemini 2.5 Flash adapter
├── controller/
│   ├── agent_controller.py    # Main orchestration
│   ├── evidence_layer.py      # Evidence bundle assembly
│   └── input_validator.py     # Input validation
├── database/
│   └── mongodb.py             # MongoDB Atlas integration
├── model/
│   └── builder.py             # Specialist model registry (optional)
├── serve/
│   └── web_server.py          # Streamlit frontend
├── tools/
│   ├── change.py              # Bi-temporal change tool
│   ├── fusion.py              # Optical+SAR fusion tool
│   ├── grounding.py           # Region grounding tool
│   └── vqa_caption.py         # VQA/captioning tool
└── utils/
    ├── image_io.py             # GeoTIFF loading + metadata
    ├── image_preprocessor.py  # Visualization + change detection
    └── report_builder.py      # PDF + Markdown export
```
