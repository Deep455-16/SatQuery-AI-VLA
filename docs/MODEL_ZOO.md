# Model Zoo

SatQuery AI is a **multi-model agentic system**, not a single checkpoint —
this is the main structural difference from GeoChat, which ships one
grounded VLM checkpoint. The controller (`satquery/controller/`) routes
each query to the appropriate specialist below, loaded lazily through
`satquery/model/builder.py::load_pretrained_model(task)`.

| Task | Specialist | Suggested open-source checkpoint |
|---|---|---|
| Single-image VQA / captioning | `vqa`, `caption` | [TerraQ-VL](https://github.com/crimsonKn1ght/TerraQ-VL) — CLIP ViT-L/14 + Qwen2.5-3B, LLaVA-style connector, trained on VRSBench |
| Text-guided region grounding | `grounding` | [GeoChat](https://github.com/mbzuai-oryx/GeoChat) — the grounded RS VLM this project is structured after |
| Bi-temporal change description / change-VQA | `change` | [VisTA](https://github.com/like413/VisTA) — unifies CDVQA + grounding, outputs text + mask |
| Optical–SAR cross-modal fusion | `fusion` | [DS_UNet](https://github.com/SebastianHafner/DS_UNet) — dual-stream U-Net fusing Sentinel-1/2 |

Until real weights are wired in, every task above returns a `MockModel`
placeholder so the full agentic pipeline (validation → routing →
execution → auditable trace) can be run and demoed end-to-end without a
GPU. Replace `MockModel` in `satquery/model/builder.py` with real
checkpoint-loading code (see the module docstring there for the exact
interface to keep: `.run(**kwargs) -> dict`).

## Swapping in a real checkpoint

```python
# satquery/model/builder.py
class TerraQVLModel:
    def __init__(self, ckpt_dir):
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(ckpt_dir)
        self.model = AutoModelForCausalLM.from_pretrained(ckpt_dir)

    def run(self, image, query, **kw):
        inputs = self.processor(images=image, text=query, return_tensors="pt")
        out = self.model.generate(**inputs)
        text = self.processor.decode(out[0], skip_special_tokens=True)
        return {"model": "TerraQ-VL", "answer": text, "confidence": 0.0}
```

Keep the `.run()` signature stable so nothing in `satquery/tools/` or
`satquery/controller/` needs to change.
