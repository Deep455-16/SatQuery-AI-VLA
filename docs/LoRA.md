# Running the demo and LoRA fine-tuning

## Running the demo

**GUI (Streamlit):**

```bash
pip install -e .
streamlit run satquery_demo.py
```

**CLI (single query, no server):**

```bash
python -m satquery.serve.cli --image scene.png \
    --query "Describe the land-cover and major objects visible in this image."

python -m satquery.serve.cli --image-a t1_optical.tif --image-b t2_optical.tif \
    --query "What changed between these two dates, and where did the change occur?"
```

## LoRA fine-tuning

Once a backbone checkpoint (TerraQ-VL / GeoChat-style connector) and a
BigEarthNet-derived instruction set are in place (see `docs/Data.md`),
launch fine-tuning the same way GeoChat does — LoRA adapters on top of a
frozen-or-lightly-tuned vision-language backbone, avoiding catastrophic
forgetting of general conversational ability while adding remote-sensing
domain knowledge:

```bash
bash scripts/finetune_lora.sh
```

Key flags (same intent as GeoChat's):

- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 backbone.
- `--mm_projector_type mlp2x_gelu`: two-layer MLP vision-language connector.
- `--image_aspect_ratio pad`: pad non-square tiles instead of cropping.
- `--lora_enable True`: LoRA rather than full fine-tuning.

DeepSpeed ZeRO configs (`scripts/zero2.json`, `scripts/zero3.json`,
`scripts/zero3_offload.json`) are provided for multi-GPU / memory-constrained
setups, same as in GeoChat.
