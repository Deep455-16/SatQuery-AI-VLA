"""
satquery/train/prepare_bigearthnet.py

Converts a BigEarthNet-style directory tree into the instruction-tuning
JSON format used by `scripts/finetune_lora.sh` / `satquery/train/train.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

try:
    import pyarrow
except ImportError:
    pass

try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    _HAS_RASTERIO = False


OPTICAL_RGB_BANDS = ("B04", "B03", "B02")  # Sentinel-2 red, green, blue
SAR_BANDS = ("VV", "VH")

# ---------------------------------------------------------------------
# Metadata and Parquet schema validation
# ---------------------------------------------------------------------

def load_metadata_parquet(parquet_path: str, split: str = 'all', max_patches: int = None) -> pd.DataFrame:
    """Load BigEarthNet metadata from parquet. Returns DataFrame."""
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load parquet file {parquet_path}: {e}")
        
    if split != 'all':
        if 'split' in df.columns:
            df = df[df['split'] == split]
        else:
            raise ValueError(f"Parquet missing 'split' column, cannot filter by {split}")
            
    if max_patches is not None:
        df = df.head(max_patches)
        
    return df

def validate_metadata_schema(df) -> list[str]:
    """Returns list of schema issues (empty if valid)."""
    issues = []
    required_cols = [
        'patch_id', 'labels', 'split', 'country', 's1_name', 's2v1_name',
        'contains_seasonal_snow', 'contains_cloud_or_shadow'
    ]
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")
    return issues

def dry_run(bigearthnet_root: str, parquet_path: str = None, max_patches: int = 10) -> dict:
    """
    Validate without full processing. Returns a summary dictionary.
    """
    root = Path(bigearthnet_root)
    
    summary = {
        'metadata_ok': True,
        'schema_issues': [],
        'patches_checked': 0,
        'patches_with_s2': 0,
        'patches_with_s1': 0,
        'patches_missing_rgb': 0,
        'splits_found': {'train': 0, 'validation': 0, 'test': 0},
        'labels_found': 0,
        'sample_labels': [],
        'warnings': [],
        'errors': []
    }
    
    # 1. Metadata check
    df = None
    if parquet_path:
        try:
            df = pd.read_parquet(parquet_path)
            issues = validate_metadata_schema(df)
            summary['schema_issues'] = issues
            if issues:
                summary['metadata_ok'] = False
            else:
                for split_name in summary['splits_found'].keys():
                    summary['splits_found'][split_name] = len(df[df['split'] == split_name])
                
                # Check sample labels
                all_labels_set = set()
                for labels in df['labels'].dropna():
                    if isinstance(labels, str):
                        labels = eval(labels)
                    all_labels_set.update(labels)
                summary['labels_found'] = len(all_labels_set)
                summary['sample_labels'] = list(all_labels_set)[:5]
                
        except Exception as e:
            summary['metadata_ok'] = False
            summary['errors'].append(f"Parquet load failed: {e}")
            
    # 2. Patch discovery
    if parquet_path and df is not None and summary['metadata_ok']:
        test_df = df.head(max_patches)
        patch_names = test_df['patch_id'].tolist()
        patch_dirs = [root / name for name in patch_names]
    else:
        try:
            patch_dirs = sorted([p for p in root.iterdir() if p.is_dir()])[:max_patches]
        except Exception as e:
            patch_dirs = []
            summary['errors'].append(f"Failed to read directory {root}: {e}")
            
    summary['patches_checked'] = len(patch_dirs)
    
    # 3. Patch validation
    for patch_dir in patch_dirs:
        if not patch_dir.exists():
            summary['errors'].append(f"Patch folder missing: {patch_dir}")
            summary['patches_missing_rgb'] += 1
            continue
            
        b2 = find_band(patch_dir, "B02")
        b3 = find_band(patch_dir, "B03")
        b4 = find_band(patch_dir, "B04")
        
        if b2 and b3 and b4:
            summary['patches_with_s2'] += 1
        else:
            summary['patches_missing_rgb'] += 1
            summary['warnings'].append(f"Missing RGB bands in {patch_dir.name}")
            
        s1_vv = find_band(patch_dir, "VV")
        s1_vh = find_band(patch_dir, "VH")
        if s1_vv or s1_vh:
            summary['patches_with_s1'] += 1
            
    return summary

# ---------------------------------------------------------------------
# Patch discovery
# ---------------------------------------------------------------------

def find_patches(bigearthnet_root: str) -> list[Path]:
    """Return one directory per BigEarthNet patch under the root."""
    root = Path(bigearthnet_root)
    if not root.is_dir():
        raise FileNotFoundError(f"BigEarthNet root not found: {bigearthnet_root}")
    patches = sorted(p for p in root.iterdir() if p.is_dir())
    if not patches:
        raise ValueError(f"No patch sub-folders found under {bigearthnet_root}")
    return patches


def load_labels(patch_dir: Path) -> list[str]:
    label_files = list(patch_dir.glob("*_labels_metadata.json"))
    if not label_files:
        return []
    with open(label_files[0]) as f:
        meta = json.load(f)
    return meta.get("labels", [])


def find_band(patch_dir: Path, band_suffix: str) -> Optional[Path]:
    matches = list(patch_dir.glob(f"*_{band_suffix}.tif")) or list(patch_dir.glob(f"*_{band_suffix}.tiff"))
    return matches[0] if matches else None


# ---------------------------------------------------------------------
# Image composition
# ---------------------------------------------------------------------

def _read_band(path: Path) -> np.ndarray:
    if _HAS_RASTERIO:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
    else:  # pragma: no cover - fallback for environments without rasterio
        arr = np.array(Image.open(path)).astype(np.float32)
    return arr


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def compose_optical_rgb(patch_dir: Path) -> Optional[np.ndarray]:
    """Stack Sentinel-2 B04/B03/B02 into an RGB uint8 array. None if bands missing."""
    band_paths = [find_band(patch_dir, b) for b in OPTICAL_RGB_BANDS]
    if any(p is None for p in band_paths):
        return None
    bands = [_normalize_to_uint8(_read_band(p)) for p in band_paths]
    return np.stack(bands, axis=-1)


def compose_sar_preview(patch_dir: Path) -> Optional[np.ndarray]:
    """Stack available Sentinel-1 VV/VH bands into a single-or-dual-channel preview."""
    band_paths = [find_band(patch_dir, b) for b in SAR_BANDS]
    available = [p for p in band_paths if p is not None]
    if not available:
        return None
    bands = [_normalize_to_uint8(_read_band(p)) for p in available]
    if len(bands) == 1:
        return bands[0]
    return np.stack(bands[:2], axis=-1)


# ---------------------------------------------------------------------
# Instruction-pair generation
# ---------------------------------------------------------------------

_CAPTION_TEMPLATES = [
    "This Sentinel-2 patch shows {labels}.",
    "The dominant land-cover types visible here are {labels}.",
    "This tile is characterised by {labels}.",
]

_VQA_TEMPLATES = [
    ("What land-cover types are present in this image?", "The image contains {labels}."),
    ("Is {probe} visible in this scene?", "{yn}, {probe} is {presence} in this scene."),
    ("Describe the land-cover and major features visible in this image.",
     "This image shows {labels}."),
]


def _format_labels(labels: list[str]) -> str:
    if not labels:
        return "an unlabeled land-cover pattern"
    if len(labels) == 1:
        return labels[0].lower()
    return ", ".join(l.lower() for l in labels[:-1]) + f", and {labels[-1].lower()}"


def build_caption_sample(sample_id: str, image_rel_path: str, labels: list[str], rng: random.Random) -> dict:
    template = rng.choice(_CAPTION_TEMPLATES)
    caption = template.format(labels=_format_labels(labels))
    return {
        "id": sample_id,
        "image": image_rel_path,
        "conversations": [
            {"from": "human", "value": "<image>\nDescribe the land-cover and major objects visible in this image."},
            {"from": "gpt", "value": caption},
        ],
    }


def build_vqa_sample(sample_id: str, image_rel_path: str, labels: list[str], all_labels_vocab: list[str], rng: random.Random) -> dict:
    q_template, a_template = rng.choice(_VQA_TEMPLATES)
    if "{probe}" in q_template:
        # 50/50 ask about a label that IS present vs one that is NOT, for balanced VQA supervision
        if labels and rng.random() < 0.5:
            probe = rng.choice(labels)
            present = True
        else:
            negatives = [l for l in all_labels_vocab if l not in labels] or ["urban fabric"]
            probe = rng.choice(negatives)
            present = False
        question = q_template.format(probe=probe.lower())
        answer = a_template.format(
            yn="Yes" if present else "No",
            probe=probe.lower(),
            presence="present" if present else "not present",
        )
    else:
        question = q_template
        answer = a_template.format(labels=_format_labels(labels))
    return {
        "id": sample_id,
        "image": image_rel_path,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }


def build_fusion_sample(sample_id: str, optical_rel_path: str, sar_rel_path: str, labels: list[str]) -> dict:
    """Cross-modal instruction sample pairing optical + SAR previews of the same patch."""
    return {
        "id": sample_id,
        "image": optical_rel_path,
        "image_pair": sar_rel_path,
        "conversations": [
            {"from": "human", "value": "<image>\n<image_pair>\nUse the optical and SAR images together to "
                                        "identify the built-up and vegetated regions."},
            {"from": "gpt", "value": f"Combining the optical and SAR evidence, this area is consistent with "
                                      f"{_format_labels(labels)}, with SAR backscatter confirming structural "
                                      f"features not fully resolved in the optical band alone."},
        ],
    }


# ---------------------------------------------------------------------
# Main conversion driver
# ---------------------------------------------------------------------

def convert(
    bigearthnet_root: str,
    output_json: str,
    output_image_dir: str,
    max_patches: Optional[int] = None,
    seed: int = 42,
    parquet_path: Optional[str] = None,
    split: str = 'all'
) -> dict:
    """
    Runs the full conversion and returns a small summary dict.
    """
    rng = random.Random(seed)
    
    root = Path(bigearthnet_root)
    if not root.is_dir():
        raise FileNotFoundError(f"BigEarthNet root not found: {bigearthnet_root}")
        
    patch_labels = {}
    patches = []
    
    if parquet_path:
        df = load_metadata_parquet(parquet_path, split=split, max_patches=max_patches)
        for _, row in df.iterrows():
            patch_name = row['patch_id']
            p = root / patch_name
            patches.append(p)
            
            # Parse labels from parquet
            labels = row['labels']
            if isinstance(labels, str):
                try:
                    labels = eval(labels)
                except:
                    labels = [labels]
            patch_labels[p] = list(labels)
    else:
        patches = find_patches(bigearthnet_root)
        if max_patches:
            patches = patches[:max_patches]
        for p in patches:
            patch_labels[p] = load_labels(p)

    os.makedirs(output_image_dir, exist_ok=True)

    # Build a global label vocabulary first (for negative sampling in VQA)
    all_labels: set[str] = set()
    for labels in patch_labels.values():
        all_labels.update(labels)
    all_labels_vocab = sorted(all_labels)

    records: list[dict] = []
    n_captions = n_vqa = n_fusion = n_skipped = 0

    for patch_dir in patches:
        labels = patch_labels.get(patch_dir, [])
        rgb = compose_optical_rgb(patch_dir)
        if rgb is None:
            n_skipped += 1
            continue

        patch_id = patch_dir.name
        optical_filename = f"{patch_id}_optical.png"
        optical_out_path = os.path.join(output_image_dir, optical_filename)
        Image.fromarray(rgb).save(optical_out_path)

        records.append(build_caption_sample(f"{patch_id}_cap", optical_filename, labels, rng))
        n_captions += 1

        records.append(build_vqa_sample(f"{patch_id}_vqa", optical_filename, labels, all_labels_vocab, rng))
        n_vqa += 1

        sar = compose_sar_preview(patch_dir)
        if sar is not None:
            sar_filename = f"{patch_id}_sar.png"
            sar_out_path = os.path.join(output_image_dir, sar_filename)
            Image.fromarray(sar).save(sar_out_path)
            records.append(build_fusion_sample(f"{patch_id}_fusion", optical_filename, sar_filename, labels))
            n_fusion += 1

    with open(output_json, "w") as f:
        json.dump(records, f, indent=2)

    summary = {
        "patches_seen": len(patches),
        "patches_skipped_missing_bands": n_skipped,
        "caption_samples": n_captions,
        "vqa_samples": n_vqa,
        "fusion_samples": n_fusion,
        "total_records": len(records),
        "output_json": output_json,
        "output_image_dir": output_image_dir,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Convert a BigEarthNet-style directory tree into SatQuery AI instruction-tuning JSON."
    )
    parser.add_argument("--bigearthnet-root", type=str, default="/path/to/BigEarthNet",
                         help="Root directory containing one sub-folder per BigEarthNet patch.")
    parser.add_argument("--metadata-parquet", type=str, default=None,
                         help="Path to metadata.parquet for label and split lookup.")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "validation", "test"],
                         help="Which split to process if using parquet.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Validate metadata and patches without processing full dataset.")
    parser.add_argument("--output-json", type=str, default="playground/data/bigearthnet_instruct.json")
    parser.add_argument("--output-image-dir", type=str, default="playground/data/bigearthnet_images")
    parser.add_argument("--max-patches", type=int, default=None,
                         help="Optional cap on number of patches processed (useful for a quick trial run).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dry_run:
        print("\n--- DRY RUN ---")
        summary = dry_run(args.bigearthnet_root, args.metadata_parquet, args.max_patches or 10)
        
        # Color codes
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        
        ok = summary['metadata_ok'] and not summary['errors']
        
        print(f"Metadata Schema: {GREEN + 'OK' if summary['metadata_ok'] else RED + 'ERRORS'}{RESET}")
        for i in summary['schema_issues']:
            print(f"  - {RED}{i}{RESET}")
            
        print(f"\nSplits Found:")
        for k, v in summary['splits_found'].items():
            print(f"  {k}: {v}")
            
        print(f"\nLabels Found: {summary['labels_found']}")
        print(f"Sample Labels: {summary['sample_labels']}")
        
        print(f"\nPatch Discovery (Checked {summary['patches_checked']}):")
        print(f"  With S2: {summary['patches_with_s2']}")
        print(f"  With S1: {summary['patches_with_s1']}")
        print(f"  Missing RGB: {summary['patches_missing_rgb']}")
        
        if summary['warnings']:
            print(f"\n{YELLOW}Warnings:{RESET}")
            for w in summary['warnings']:
                print(f"  - {w}")
                
        if summary['errors']:
            print(f"\n{RED}Errors:{RESET}")
            for e in summary['errors']:
                print(f"  - {e}")
                
        if not ok:
            print(f"\n{RED}Dry run FAILED.{RESET}")
            sys.exit(1)
        else:
            print(f"\n{GREEN}Dry run PASSED.{RESET}")
            sys.exit(0)

    # Actual conversion
    try:
        summary = convert(
            bigearthnet_root=args.bigearthnet_root,
            output_json=args.output_json,
            output_image_dir=args.output_image_dir,
            max_patches=args.max_patches,
            seed=args.seed,
            parquet_path=args.metadata_parquet,
            split=args.split
        )
        print(json.dumps(summary, indent=2))
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
