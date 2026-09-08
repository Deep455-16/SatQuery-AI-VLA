"""
Tests for satquery/train/prepare_bigearthnet.py.

Builds a small synthetic BigEarthNet-like directory tree (real GeoTIFF
band files + real labels_metadata.json, just fabricated pixel data), runs
the actual converter against it, and then feeds the converter's own
output images back through satquery.utils.image_io.load_image and the
AgentController to confirm the generated data is genuinely usable by the
rest of the backend, not just structurally shaped like it.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import rasterio
from rasterio.transform import from_origin

from satquery.train.prepare_bigearthnet import convert
from satquery.utils.image_io import load_image
from satquery.controller.agent_controller import AgentController


def _write_band(path, size=32, seed=0):
    rng = np.random.RandomState(seed)
    arr = (rng.rand(size, size) * 3000).astype(np.uint16)
    transform = from_origin(0, 0, 10, 10)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype=arr.dtype, crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(arr, 1)


def _make_synthetic_bigearthnet(root, n_patches=3, with_sar=True):
    for i in range(n_patches):
        patch_id = f"S2A_MSIL2A_TEST_patch_{i:04d}"
        patch_dir = os.path.join(root, patch_id)
        os.makedirs(patch_dir, exist_ok=True)

        for band, seed in zip(("B02", "B03", "B04"), (1, 2, 3)):
            _write_band(os.path.join(patch_dir, f"{patch_id}_{band}.tif"), seed=seed + i)

        if with_sar and i % 2 == 0:
            for band, seed in zip(("VV", "VH"), (10, 11)):
                _write_band(os.path.join(patch_dir, f"{patch_id}_{band}.tif"), seed=seed + i)

        labels = ["Pastures", "Coniferous forest"] if i % 2 == 0 else ["Urban fabric"]
        with open(os.path.join(patch_dir, f"{patch_id}_labels_metadata.json"), "w") as f:
            json.dump({"labels": labels}, f)


def test_converter_produces_valid_instruct_json(tmp_path):
    root = tmp_path / "bigearthnet_root"
    out_json = tmp_path / "bigearthnet_instruct.json"
    out_images = tmp_path / "bigearthnet_images"
    _make_synthetic_bigearthnet(str(root), n_patches=4, with_sar=True)

    summary = convert(
        bigearthnet_root=str(root),
        output_json=str(out_json),
        output_image_dir=str(out_images),
    )

    assert summary["patches_seen"] == 4
    assert summary["caption_samples"] == 4
    assert summary["vqa_samples"] == 4
    assert summary["fusion_samples"] == 2  # every other patch has SAR bands
    assert summary["total_records"] == 4 + 4 + 2

    assert out_json.exists()
    with open(out_json) as f:
        records = json.load(f)
    assert len(records) == summary["total_records"]

    for rec in records:
        assert "conversations" in rec and len(rec["conversations"]) == 2
        assert rec["conversations"][0]["from"] == "human"
        assert "<image>" in rec["conversations"][0]["value"]
        assert os.path.exists(out_images / rec["image"])


def test_converted_images_are_loadable_by_backend(tmp_path):
    """Confirms the converter's output is genuinely wired to the rest of the
    pipeline: images it writes load through satquery.utils.image_io and
    can be fed straight into the AgentController."""
    root = tmp_path / "bigearthnet_root"
    out_json = tmp_path / "bigearthnet_instruct.json"
    out_images = tmp_path / "bigearthnet_images"
    _make_synthetic_bigearthnet(str(root), n_patches=2, with_sar=False)

    summary = convert(
        bigearthnet_root=str(root),
        output_json=str(out_json),
        output_image_dir=str(out_images),
    )
    assert summary["total_records"] > 0

    with open(out_json) as f:
        records = json.load(f)

    controller = AgentController()
    for rec in records:
        image_path = str(out_images / rec["image"])
        arr, meta = load_image(image_path)
        assert meta.modality in ("optical", "sar", "unknown")

        question = rec["conversations"][0]["value"].replace("<image>\n", "")
        result = controller.handle_query(question, [(arr, meta)]).to_dict()
        assert result["task"] != "input_validation_failed"
        # New result structure uses evidence_strength instead of confidence
        assert "evidence_strength" in result
        assert result["evidence_strength"] in ("high", "moderate", "limited", "")
