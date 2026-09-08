"""
satquery/train/prepare_bigearthnet_csv.py

Converts data/metadata.csv (BigEarthNet) into an instruction-tuning JSONL
file for LoRA fine-tuning of TerraQ-VL (LLaVA-1.5-7B).

Since only metadata is available (no actual .tif patch images),
this generates TEXT-ONLY instruction pairs that inject BigEarthNet
land-cover taxonomy and satellite vocabulary into the model.

Usage:
    python -m satquery.train.prepare_bigearthnet_csv \\
        --csv_path data/metadata.csv \\
        --out data/bigearthnet_instructions.jsonl \\
        --max_patches 5000
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
from pathlib import Path

# ── Label parsing ────────────────────────────────────────────────────────────

def parse_labels(raw: str) -> list[str]:
    """Parse BigEarthNet label strings like \"['Arable land' 'Mixed forest']\"."""
    if not raw or not isinstance(raw, str):
        return []
    raw = raw.strip()
    # Try JSON first
    try:
        parsed = json.loads(raw.replace("'", '"'))
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if x]
    except Exception:
        pass
    # Extract quoted strings
    matches = re.findall(r"'([^']+)'|\"([^\"]+)\"", raw)
    labels = [m[0] or m[1] for m in matches if (m[0] or m[1]).strip()]
    if labels:
        return labels
    # Fallback: split by space-separated words after bracket removal
    cleaned = raw.strip("[]").replace("'", "").replace('"', "")
    return [x.strip() for x in cleaned.split(",") if x.strip()]


# ── Label → description mappings ────────────────────────────────────────────

WATER_LABELS = {
    "Inland waters", "Sea and ocean", "Coastal lagoons",
    "Estuaries", "Water bodies",
}
FOREST_LABELS = {
    "Broad-leaved forest", "Coniferous forest", "Mixed forest",
    "Transitional woodland, shrub",
}
VEGETATION_LABELS = {
    "Pastures", "Arable land", "Permanently irrigated land",
    "Rice fields", "Vineyards", "Fruit trees and berry plantations",
    "Olive groves", "Annual crops associated with permanent crops",
    "Complex cultivation patterns", "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
}
URBAN_LABELS = {
    "Continuous urban fabric", "Discontinuous urban fabric",
    "Industrial or commercial units", "Road and rail networks and associated land",
    "Port areas", "Airports", "Mineral extraction sites",
    "Construction sites", "Green urban areas", "Sport and leisure facilities",
}


def has_water(labels: list[str]) -> bool:
    return bool(set(labels) & WATER_LABELS)

def has_forest(labels: list[str]) -> bool:
    return bool(set(labels) & FOREST_LABELS)

def has_agriculture(labels: list[str]) -> bool:
    return bool(set(labels) & VEGETATION_LABELS)

def has_urban(labels: list[str]) -> bool:
    return bool(set(labels) & URBAN_LABELS)

def dominant_class(labels: list[str]) -> str:
    if has_urban(labels):
        return "urban/industrial"
    if has_agriculture(labels):
        return "agricultural"
    if has_forest(labels):
        return "forested"
    if has_water(labels):
        return "water-dominated"
    return "mixed land cover"


# ── Instruction templates ────────────────────────────────────────────────────

def build_instructions(row: dict) -> list[dict]:
    """Build 4–6 instruction pairs from one BigEarthNet metadata row."""
    patch_id = row.get("patch_id", "UNKNOWN")
    labels   = parse_labels(row.get("labels", ""))
    country  = row.get("country", "unknown country")
    snow     = str(row.get("contains_seasonal_snow", "False")).lower() == "true"
    cloud    = str(row.get("contains_cloud_or_shadow", "False")).lower() == "true"

    if not labels:
        return []

    label_str = ", ".join(labels)
    dom = dominant_class(labels)
    water = has_water(labels)
    forest = has_forest(labels)
    agri   = has_agriculture(labels)
    urban  = has_urban(labels)

    instructions = []

    # 1. Land-cover classes
    instructions.append({
        "conversations": [
            {"role": "user", "value": f"What land-cover classes are present in this Sentinel-2 satellite patch from {country}?"},
            {"role": "assistant", "value": f"This patch contains the following land-cover classes: {label_str}. The scene is primarily {dom}."},
        ]
    })

    # 2. Water visibility
    if water:
        instructions.append({
            "conversations": [
                {"role": "user", "value": "Is there water visible in this satellite image?"},
                {"role": "assistant", "value": f"Yes, water bodies are present in this scene. The patch contains: {', '.join(l for l in labels if l in WATER_LABELS)}."},
            ]
        })
    else:
        instructions.append({
            "conversations": [
                {"role": "user", "value": "Is there water visible in this satellite image?"},
                {"role": "assistant", "value": "No water bodies are detected in this patch. The land cover is dominated by " + dom + "."},
            ]
        })

    # 3. Vegetation
    if forest or agri:
        veg_types = [l for l in labels if l in FOREST_LABELS | VEGETATION_LABELS]
        instructions.append({
            "conversations": [
                {"role": "user", "value": "Describe the vegetation present in this satellite image."},
                {"role": "assistant", "value": f"The patch contains vegetation including: {', '.join(veg_types)}. "
                    + ("Forested areas are visible with tree canopy cover. " if forest else "")
                    + ("Agricultural fields indicate cultivated land use. " if agri else "")},
            ]
        })

    # 4. Cloud / shadow
    cloud_ans = (
        "Yes, this patch contains cloud cover or shadow, which may obscure surface features."
        if cloud else
        "No significant cloud cover or shadow is present, providing clear visibility of the surface."
    )
    instructions.append({
        "conversations": [
            {"role": "user", "value": "Does this satellite image contain clouds or shadows?"},
            {"role": "assistant", "value": cloud_ans},
        ]
    })

    # 5. General scene description
    desc_parts = []
    if urban:    desc_parts.append("urban and built-up areas")
    if agri:     desc_parts.append("agricultural fields")
    if forest:   desc_parts.append("forested zones")
    if water:    desc_parts.append("water bodies")
    if snow:     desc_parts.append("seasonal snow")
    description = ", ".join(desc_parts) if desc_parts else dom

    instructions.append({
        "conversations": [
            {"role": "user", "value": "Describe the land-cover of this satellite image."},
            {"role": "assistant", "value": f"This Sentinel-2 patch from {country} shows {description}. "
                f"The classified land-cover includes: {label_str}."},
        ]
    })

    # 6. Country / location
    instructions.append({
        "conversations": [
            {"role": "user", "value": "Which country does this satellite patch come from?"},
            {"role": "assistant", "value": f"This satellite patch was acquired over {country}. "
                f"The patch ID is {patch_id}."},
        ]
    })

    return instructions


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare BigEarthNet metadata.csv as LoRA instruction JSONL")
    parser.add_argument("--csv_path",    type=str, default="data/metadata.csv")
    parser.add_argument("--out",         type=str, default="data/bigearthnet_instructions.jsonl")
    parser.add_argument("--max_patches", type=int, default=10000, help="Max patches to process (0=all)")
    parser.add_argument("--split",       type=str, default="all",  help="train|val|test|all")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not installed. Run: pip install pandas")
        sys.exit(1)

    if not os.path.exists(args.csv_path):
        print(f"ERROR: CSV not found at {args.csv_path}")
        sys.exit(1)

    print(f"Loading {args.csv_path} …")
    df = pd.read_csv(args.csv_path)
    print(f"  Loaded {len(df):,} patches.")

    if args.split != "all" and "split" in df.columns:
        df = df[df["split"] == args.split]
        print(f"  Filtered to split='{args.split}': {len(df):,} patches.")

    if args.max_patches and args.max_patches > 0:
        df = df.sample(min(args.max_patches, len(df)), random_state=args.seed)
        print(f"  Sampled {len(df):,} patches.")

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    total_written = 0
    skipped = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for _, row in df.iterrows():
            instructions = build_instructions(row.to_dict())
            if not instructions:
                skipped += 1
                continue
            for inst in instructions:
                fout.write(json.dumps(inst, ensure_ascii=False) + "\n")
                total_written += 1

    print(f"\nDone! Written {total_written:,} instruction pairs to {args.out}")
    print(f"Skipped {skipped} patches with empty labels.")
    print(f"\nNext step — run LoRA fine-tuning:")
    print(f"  python -m satquery.train.train \\")
    print(f"    --model_name_or_path llava-hf/llava-1.5-7b-hf \\")
    print(f"    --data_path {args.out} \\")
    print(f"    --lora_enable \\")
    print(f"    --output_dir checkpoints/terraq-vl-bigearthnet-lora")


if __name__ == "__main__":
    main()
