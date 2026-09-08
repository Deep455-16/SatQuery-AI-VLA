# SatQuery AI 🛰️: An Agentic Vision-Language Assistant for Multimodal Remote-Sensing Image Analysis

<p align="center">
  <em>Built for ISRO/SAC Problem Statement 26167 — Smart India Hackathon</em>
</p>

[![Problem Statement](https://img.shields.io/badge/SIH-PS%2026167-87CEEB)](#)
[![Organization](https://img.shields.io/badge/ISRO-Dept.%20of%20Space-F9D371)](#)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](#)

---

## 📢 Status

- Agentic controller, input validator, specialist-tool wrappers, Streamlit
  GUI, CLI demo, and evaluation scripts are implemented and passing smoke
  tests.
- Specialist models are currently **mocked** (`MockModel` in
  `satquery/model/builder.py`) so the whole pipeline runs without a GPU or
  downloaded weights — see [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md) for the
  real checkpoints to plug in next.
- Fine-tuning on BigEarthNet has not yet been run; `satquery/train/train.py`
  and `scripts/finetune_lora.sh` are wired up and ready for a backbone
  checkpoint.

---

## 🛰️ Overview

SatQuery AI is an **agentic**, query-driven vision-language assistant for
remote-sensing image analysis. Where a general-purpose VLM (or a single
grounded RS-VLM like GeoChat) answers directly from one image and a task
token, SatQuery AI sits a **controller** in front of a registry of
remote-sensing specialist models. Given a natural-language query and one
or two input images, the controller:

1. classifies the input configuration — a single image, a co-registered
   optical + SAR pair, or a bi-temporal pair;
2. checks modality, format, and geometric compatibility;
3. classifies the requested task from the query;
4. selects and executes the matching specialist tool(s);
5. returns an evidence-grounded answer with a confidence score; and
6. logs an auditable execution trace (task, tool/model, parameters,
   outputs) — no internal chain-of-thought is exposed, matching the
   evaluation rubric.

This differs from GeoChat's architecture (a single LLaVA-style grounded
VLM answering everything through task tokens in one conversation) by
routing across **multiple specialist models**, and by mandatorily
supporting **multi-image** inputs — bi-temporal pairs for change analysis
and cross-modal optical–SAR pairs for joint information extraction —
which a single-image VLM does not natively handle.

---

## 🌍 Applications & Use Cases

SatQuery AI can be deployed across numerous critical domains:
- **Disaster Management & Response:** Rapid assessment of flood extents, earthquake damage, and forest fires using bi-temporal change detection and SAR imagery.
- **Urban Planning & Infrastructure:** Monitoring urban sprawl, construction progress, and land-use classification.
- **Agriculture & Environment:** Tracking deforestation, crop health monitoring, and water resource management.
- **Defense & Security:** Strategic monitoring, object detection (ships, aircraft), and border surveillance using fused Optical-SAR data.

## 🚀 Numerous Advantages

- **Multi-Modal Capabilities:** Natively handles Optical, SAR, and fused imagery.
- **Agentic Routing:** Dynamically selects the most efficient and accurate subagent for the specific task, reducing hallucinations and improving response times.
- **Real-Time Processing:** Incorporates fast fallback models and heuristic routing to ensure real-time query resolution even on constrained hardware.
- **Cloud-Local Hybrid Resilience:** Leverages local models (TeraQ-VLM) as standby, ensuring operations continue even when API limits or network issues arise.

---

## Contents
- [Install](#install)
- [Model Zoo](docs/MODEL_ZOO.md)
- [Data](docs/Data.md)
- [Demo](#satquery-ai-weights-and-demo)
- [Train](#train)
- [Evaluation](docs/Evaluation.md)

## Install

1. Clone this repository and navigate to the SatQuery AI folder
```bash
git clone https://github.com/your-org/satquery-ai.git
cd satquery-ai
```

2. Install the package
```bash
conda create -n satquery python=3.10 -y
conda activate satquery
pip install --upgrade pip
pip install -e .
```

3. Install additional packages for training / specialist-model inference
```bash
pip install -e ".[train]"
```

4. Environment Setup & Execution
```bash
cp .env.example .env
# Configure your Gemini API Key in the .env file.
# To start the backend API and React frontend servers:
./start_app.bat
```

---

## SatQuery AI: Weights and Demo

Until real specialist checkpoints are wired in (see
[Model Zoo](docs/MODEL_ZOO.md)), every task is served by a deterministic
mock model so the full agentic loop can be exercised end-to-end.

**GUI:**
```bash
streamlit run satquery_demo.py
```

**CLI:**
```bash
python -m satquery.serve.cli --image scene.png \
    --query "Describe the land-cover and major objects visible in this image."
```

See [docs/LoRA.md](docs/LoRA.md) for demo and fine-tuning instructions in
full.

---

## Train

SatQuery AI's remote-sensing adaptation uses **BigEarthNet** — co-registered
Sentinel-1 SAR and Sentinel-2 multispectral imagery paired with text
annotations — as the mandatory fine-tuning dataset, per the problem
statement.

`satquery/train/prepare_bigearthnet.py` converts a real BigEarthNet
download into the instruction-tuning JSON `scripts/finetune_lora.sh`
expects — reading actual GeoTIFF bands and label metadata, composing
optical/SAR preview images, and emitting captioning, VQA, and
cross-modal fusion instruction samples:

```bash
python -m satquery.train.prepare_bigearthnet \
    --bigearthnet-root /path/to/BigEarthNet \
    --output-json playground/data/bigearthnet_instruct.json \
    --output-image-dir playground/data/bigearthnet_images
```

This converter is covered by `tests/test_data_prep.py`, which runs it
against a synthetic BigEarthNet-like tree and confirms the output loads
correctly through `satquery.utils.image_io` and the `AgentController` —
see [docs/Data.md](docs/Data.md) for the full input layout and options.

### Hyperparameters (per specialist backbone)

| Backbone | Global Batch Size | Learning rate | Epochs | Max length |
| --- | ---: | ---: | ---: | ---: |
| VQA / caption (TerraQ-VL) | 32 | 2e-4 (LoRA) | 1 | 2048 |
| Grounding (GeoChat-style) | 32 | 2e-4 (LoRA) | 1 | 2048 |
| Change-VQA (VisTA) | 16 | 1e-4 | 1 | 1024 |
| Optical-SAR fusion (DS_UNet) | 16 | 1e-3 | 30 | n/a |

### LoRA fine-tuning

```bash
bash scripts/finetune_lora.sh
```

Key flags:
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px backbone.
- `--mm_projector_type mlp2x_gelu`: two-layer MLP vision-language connector.
- `--image_aspect_ratio pad`: pads non-square tiles instead of cropping, reducing hallucination on partial scenes.
- `--lora_enable True`: LoRA adapters rather than full fine-tuning, preserving general conversational ability.

DeepSpeed ZeRO-2/3 configs are included under `scripts/` for multi-GPU
training.

---

## Evaluation

SatQuery AI is evaluated against the prescribed public benchmark test
splits — **VRSBench** (captioning, grounding, VQA), **RSVQA**
(single-image VQA), and **CDVQA** (change-based VQA) — plus a held-out
ISRO/SAC evaluation set of co-registered Cartosat-2S optical and RISAT
SAR pairs. Scores are normalised before combining across metrics. We
evaluate with greedy decoding, matching the real-time behaviour of the
CLI/GUI demo. Full instructions: [docs/Evaluation.md](docs/Evaluation.md).

```bash
python -m satquery.eval.batch_satquery_vqa       --questions-file rsvqa_test_questions.json --answers-file rsvqa_predictions.jsonl
python -m satquery.eval.batch_satquery_grounding --questions-file vrsbench_grounding.json   --answers-file grounding_predictions.jsonl
python -m satquery.eval.batch_satquery_change    --questions-file cdvqa_test.json           --answers-file change_predictions.jsonl
```

---

## 🏆 Key Differences from a Single-Image RS-VLM (e.g. GeoChat)

- **Agentic orchestration, not one model.** A controller (`satquery/controller/agent_controller.py`)
  interprets the query, classifies the task, and dispatches to whichever
  specialist tool fits — VQA/captioning, grounding, change-VQA, or
  optical–SAR fusion — instead of one model answering everything.
- **Multi-image input handling is mandatory, not incidental.** The input
  validator (`satquery/controller/input_validator.py`) explicitly
  classifies single/cross-modal/bi-temporal configurations and checks
  dimension and CRS compatibility before any model runs.
- **Bi-temporal change reasoning.** A dedicated change-analysis tool
  (backed by a VisTA-style model) handles change description and
  change-VQA — a task a single-image VLM cannot perform without a second
  image slot and temporal reasoning head.
- **Optical–SAR joint analysis.** A dedicated fusion tool (backed by a
  DS_UNet-style dual-stream encoder) extracts complementary structural
  (SAR) and spectral (optical) information from co-registered pairs.
- **Auditable execution trace.** Every response ships with the selected
  task, tool/model name, permitted parameters, and confidence — matching
  the problem statement's evaluation rubric of observable trace over
  internal reasoning.

---

### 🖼️ Architecture & Model Integration

**Note:** *We have currently integrated 7 models, but currently 3 are functional and performing to their full extent. Also we had integrated the TeraQ-VLM model as standby which performs the work of every single subagent from SAR to image fetching and generating answers in real time. Apart from that we had used the Gemini 3.5-Flash as a relevance model in order to check the relevance of the answer given by the individual subagent based on the image and type of question asked.*

### Integrated Subagents and Models
- **Intent Classifier** — Qwen2.5-0.5B-Instruct
- **Captioning Agent** — BLIP-2 (OPT-2.7B)
- **Change-Understanding Agent** — Change-Agent (LLaVA-style) / fallback: Qwen2.5-VL-3B-Instruct
- **Optical–SAR Fusion Agent** — U-Net/Siamese ResNet (custom CNN, no pretrained LLM)
- **Input Validator** — None (rule-based, no model)
- **VQA Agent** — Qwen2.5-VL-3B-Instruct
- **Evidence Aggregator** — Qwen2.5-0.5B-Instruct
- **Base model (shared backbone):** Qwen2.5-VL-3B-Instruct

```text
                     ┌──────────────────────────────┐
   query + image(s)  │      Streamlit / React UI    │
  ───────────────────▶       (satquery.serve)       │
                     └─────────────┬────────────────┘
                                   │
                     ┌─────────────▼────────────────┐
                     │      Input Validator         │ (Rule-based)
                     │ (controller.input_validator) │ modality / format
                     └─────────────┬────────────────┘
                                   │ classified input config
                     ┌─────────────▼────────────────┐
                     │      Agent Controller        │ 
                     │ (Intent: Qwen2.5-0.5B-Inst)  │ query → task routing
                     └──┬──────────┬──────────┬─────┘
           single-image │          │          │ cross-modal
                        │          │ bi-temp  │
         ┌──────────────▼─┐ ┌──────▼───────┐ ┌▼──────────────┐
         │ VQA / Caption  │ │ Change Agent │ │ Fusion Agent  │
         │ (Qwen2.5-VL-3B/│ │ (LLaVA-style/│ │ (U-Net/ResNet)│
         │  BLIP-2)       │ │ Qwen fallback│ │               │
         └───────┬────────┘ └──────┬───────┘ └───────┬───────┘
                 │                 │                 │
                 └─────────┬───────┴─────────┬───────┘
                           │                 │
                   [Gemini 3.5-Flash (Relevance Checker)]
                   [TeraQ-VLM (Standby/Fallback Engine)]
                           │
                     ┌─────▼────────────────────────┐
                     │    Evidence Aggregator       │ (Qwen2.5-0.5B-Inst)
                     │    + Execution Trace         │
                     └──────────────────────────────┘
```

Place architecture/qualitative-result screenshots under `demo_images/`
as the specialist models are wired in and evaluated, mirroring GeoChat's
`images/` gallery of scene-classification, VQA, and grounding examples.

---

## 📜 Citation

If specialist models from the Model Zoo are used, please cite their
original works (GeoChat, TerraQ-VL/VRSBench, VisTA, DS_UNet — see
[docs/MODEL_ZOO.md](docs/MODEL_ZOO.md) for links). This scaffold itself
can be cited as:

```bibtex
@misc{satqueryai2026,
  title  = {SatQuery AI: An Agentic Vision-Language Assistant for Multimodal Remote-Sensing Image Analysis},
  note   = {Developed for ISRO/SAC Smart India Hackathon Problem Statement 26167},
  year   = {2026}
}
```


