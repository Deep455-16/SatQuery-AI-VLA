# SatQuery AI — System Architecture
## Problem Statement 26167 — ISRO/SAC Smart India Hackathon 2026

## Architecture Diagram

```
+-------------------------------------------------------------+
|                     User Interface                          |
|         (Web App / Streamlit / API / Command Line)          |
+-----------------------------+-------------------------------+
                              |
                     +--------v--------+
                     | Input Validator & |
                     | Config Builder    |
                     +--------+--------+
                              |
                     +--------v--------+
                     | Agent Controller  |
                     | (Task Classifier) |
                     +--------+--------+
                              |
    +-------------------------+-------------------------+
    |                         |                         |
+---v---+                 +---v---+                 +---v---+
| VQA / |                 |Change |                 |Optical|
|Caption|                 | Detect|                 |  SAR  |
| (VLM) |                 |       |                 |Fusion |
+---+---+                 +---+---+                 +---+---+
    |                         |                         |
    +-------------------------+-------------------------+
                              |
                     +--------v--------+
                     |  Evidence Layer   |
                     | (XAI / Heatmaps)  |
                     +--------+--------+
                              |
                     +--------v--------+
                     | Report Generator  |
                     | (JSON / MD / PDF) |
                     +-----------------+
```

## Component Descriptions
- **User Interface**: Accepts user inputs including multi-image uploads and natural language queries. 
- **Input Validator**: Ensures dimensions match (when appropriate), validates CRSs, extracts metadata, and categorizes input as SINGLE, BITEMPORAL, CROSS_MODAL, or MULTI_IMAGE.
- **Agent Controller**: The orchestrator that takes the validated input and query, formally categorizes the task, and dispatches to the corresponding specialist tool.
- **VQA / Caption**: Utilizes TerraQ-VL and Gemini to produce textual answers and groundings.
- **Change Detect**: Analyzes differences in bitemporal pairs, producing change heatmaps and semantic interpretation.
- **Optical-SAR Fusion**: Synthesizes features from cross-modal data to answer queries that require both.
- **Evidence Layer**: Provides explainability by attaching visual groundings (heatmaps, RGB visualizations) and confidence scores to the results.
- **Report Generator**: Formats the `ExecutionTrace` into user-friendly JSON, Markdown, or PDF formats.

## Data Flow
1. **Input**: User provides N images and a query string.
2. **Validation**: `InputValidator` checks sizes, CRS, modalities.
3. **Dispatch**: `AgentController` selects a pipeline branch based on image counts and query intent.
4. **Execution**: The specialized module interacts with the VLM/LLM to generate inferences.
5. **Trace Assembly**: Raw inferences are collected into an `ExecutionResult` alongside metadata.
6. **Output Generation**: `EvidenceBuilder` generates necessary images, `ReportBuilder` creates the final document.

## Technology Stack
| Component | Technology |
|---|---|
| Core Language | Python 3.10+ |
| VLM | HuggingFace Transformers, PEFT (LoRA) |
| Reasoning | Google Gemini 2.5 Flash |
| Image Processing | Rasterio, OpenCV, Pillow, NumPy |
| Web Backend | Flask |
| Frontend | React, Streamlit |
| Data | Pandas, PyArrow |

## Task Classification
| Input Type | Supported Tasks |
|---|---|
| SINGLE_IMAGE | VQA, CAPTION, GROUNDING |
| BITEMPORAL_PAIR | CHANGE_DETECTION, CHANGE_VQA |
| CROSS_MODAL_PAIR | OPTICAL_SAR_FUSION |
| MULTI_IMAGE | COMPARISON |

## Specialist Registry
| Tool | Purpose |
|---|---|
| `vqa_caption` | Single image QA and general descriptions |
| `grounding` | Identifying specific locations of objects in a single image |
| `change` | Bitemporal change analysis and metrics |
| `fusion` | Cross-modal feature extraction |

## Input Type Support
| Format | Modalities | Requirements |
|---|---|---|
| GeoTIFF | Optical, SAR | CRS metadata extracted if available |
| PNG / JPG | Visual | Assumed non-georeferenced |

## API Endpoints
| Endpoint | Method | Purpose |
|---|---|---|
| `/api/analyze` | POST | Submits query and images for agent processing |
| `/api/health` | GET | System status and VLM/LLM availability |

## Hardware Requirements
- **Inference (Gemini only)**: Minimal. Any standard modern CPU, 4GB RAM.
- **Inference (TerraQ-VL Local)**: NVIDIA GPU (8GB+ VRAM recommended) or CPU (slower).
- **Training (LoRA)**: NVIDIA GPU (16GB+ VRAM recommended), 16GB+ System RAM.
