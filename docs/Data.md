# Data

## Adaptation dataset (mandatory)

**BigEarthNet** is the primary dataset for adapting image–text
representations to multisensor remote-sensing data, as required by the
problem statement. It provides co-registered Sentinel-1 SAR and
Sentinel-2 multispectral imagery with diverse text annotations.

- Link: https://arxiv.org/abs/2603.29630 (open source)

### Converting BigEarthNet into instruction-tuning JSON

`satquery/train/prepare_bigearthnet.py` is a **working converter** — it
reads real Sentinel-2 (and, where present, Sentinel-1) GeoTIFF band
files plus each patch's `*_labels_metadata.json`, and emits the same
`{"id", "image", "conversations": [...]}` instruction shape GeoChat uses
for `GeoChat_Instruct.json`. It expects the standard on-disk BigEarthNet
layout — one sub-folder per patch:

```
bigearthnet_root/
  S2A_MSIL2A_..._patch_0001/
    S2A_MSIL2A_..._patch_0001_B02.tif        # Sentinel-2 blue
    S2A_MSIL2A_..._patch_0001_B03.tif        # Sentinel-2 green
    S2A_MSIL2A_..._patch_0001_B04.tif        # Sentinel-2 red
    S2A_MSIL2A_..._patch_0001_VV.tif         # optional Sentinel-1 SAR
    S2A_MSIL2A_..._patch_0001_VH.tif         # optional Sentinel-1 SAR
    S2A_MSIL2A_..._patch_0001_labels_metadata.json
  S2A_MSIL2A_..._patch_0002/
    ...
```

Run it directly on a BigEarthNet download:

```bash
python -m satquery.train.prepare_bigearthnet \
    --bigearthnet-root /path/to/BigEarthNet \
    --output-json playground/data/bigearthnet_instruct.json \
    --output-image-dir playground/data/bigearthnet_images \
    --max-patches 5000   # optional cap for a quick trial run
```

For each patch it:
1. composes an RGB preview from the B04/B03/B02 bands and writes it to
   `--output-image-dir`;
2. emits a **captioning** sample from the patch's label list;
3. emits a **VQA** sample (mixing "what land-cover is present" questions
   with balanced yes/no presence questions against a global label
   vocabulary built across the whole dataset);
4. if Sentinel-1 VV/VH bands are present for that patch, also composes a
   SAR preview and emits a **cross-modal fusion** instruction sample
   pairing the two.

It prints a JSON summary (patches seen/skipped, sample counts per task)
so you can sanity-check a run before pointing training at it. The
converter is exercised end-to-end in `tests/test_data_prep.py`, which
builds a small synthetic BigEarthNet-like tree, runs the real conversion
code against it, then feeds the generated PNGs back through
`satquery.utils.image_io.load_image` and `AgentController` to confirm
the output is directly usable by the rest of the backend — not just
structurally similar to GeoChat's format.

Once generated, point `scripts/finetune_lora.sh` at the resulting
`bigearthnet_instruct.json` and `bigearthnet_images/` folder (already
wired as the script's defaults).

## Evaluation benchmarks (prescribed)

| Benchmark | Task | Used for |
|---|---|---|
| [VRSBench](https://github.com/lx709/VRSBench) | Captioning, grounding, VQA | Single-image evaluation |
| RSVQA | Visual question answering | Single-image VQA evaluation |
| CDVQA | Change-based VQA | Multitemporal change evaluation |

Batch-prediction scripts for each are provided under `satquery/eval/`,
mirroring GeoChat's `batch_geochat_*.py` scripts:

```bash
python -m satquery.eval.batch_satquery_vqa \
    --questions-file rsvqa_test_questions.json \
    --answers-file rsvqa_predictions.jsonl

python -m satquery.eval.batch_satquery_grounding \
    --questions-file vrsbench_grounding.json \
    --answers-file grounding_predictions.jsonl

python -m satquery.eval.batch_satquery_change \
    --questions-file cdvqa_test.json \
    --answers-file change_predictions.jsonl
```

## Final ISRO/SAC evaluation set

Final scoring also uses a held-out ISRO/SAC dataset of pre-georeferenced,
co-registered **Cartosat-2S optical** and **RISAT SAR** image pairs with
task-specific reference answers/labels/masks. Annotations for this set are
not disclosed to participating teams — build your validation split from
BigEarthNet/VRSBench/RSVQA/CDVQA only.

## Supported input formats

| Input type | Formats |
|---|---|
| Single image | GeoTIFF / TIFF (geospatial); PNG / JPEG (benchmark datasets only) |
| Cross-modal pair | Co-registered optical/multispectral + SAR, same formats |
| Bi-temporal pair | Two spatially corresponding images, same formats |

`satquery/utils/image_io.py` handles loading and modality/compatibility
checks for all of the above.
