"""
SatQuery AI — Demonstration Dataset Guide

This file describes how to obtain legally free satellite images suitable
for demonstrating SatQuery AI at SIH. It does NOT download anything automatically.

IMPORTANT: Do NOT commit actual satellite images to this Git repository.
           Large GeoTIFFs should be stored locally and listed in .gitignore.
"""
import urllib.request
import os
import json

DATASETS = {
    "sentinel2_copernicus": {
        "name": "Sentinel-2 Optical (ESA Copernicus)",
        "url": "https://dataspace.copernicus.eu/",
        "license": "Free for all users under Copernicus Data Policy",
        "format": "GeoTIFF (SAFE format)",
        "resolution": "10m (RGB+NIR), 20m (SWIR)",
        "description": "Multispectral optical satellite imagery. Best for: vegetation, water, urban analysis.",
        "note": "Create a free account at dataspace.copernicus.eu to browse and download.",
    },
    "sentinel1_sar_copernicus": {
        "name": "Sentinel-1 SAR (ESA Copernicus)",
        "url": "https://dataspace.copernicus.eu/",
        "license": "Free for all users under Copernicus Data Policy",
        "format": "GeoTIFF (SAFE format)",
        "resolution": "10m (IW mode)",
        "description": "SAR imagery (VV/VH polarization). Best for: flood mapping, ship detection, change detection.",
        "note": "Use the Copernicus Data Space Ecosystem browser to download co-registered pairs.",
    },
    "landsat_usgs": {
        "name": "Landsat 8/9 (USGS Earth Explorer)",
        "url": "https://earthexplorer.usgs.gov/",
        "license": "Public Domain (USGS)",
        "format": "GeoTIFF",
        "resolution": "30m (multispectral), 15m (pan)",
        "description": "Long time-series optical imagery. Best for: change detection (decades), land cover.",
        "note": "Free account at earthexplorer.usgs.gov. Download individual bands as GeoTIFF.",
    },
    "cartosat_isro": {
        "name": "Cartosat-2S (ISRO Bhuvan)",
        "url": "https://bhuvan.nrsc.gov.in/",
        "license": "Free for research under NRSC/ISRO policy",
        "format": "GeoTIFF",
        "resolution": "0.65m (PAN)",
        "description": "High-resolution panchromatic imagery from ISRO. Ideal for SIH ISRO context.",
        "note": "Access via Bhuvan portal (bhuvan.nrsc.gov.in). Registration required.",
    },
}

DEMO_SCENARIOS = [
    {
        "scenario": "Urban Area",
        "description": "Dense city center with roads, buildings, parks",
        "modality": "optical",
        "suggested_query": "Describe the land-cover features and urban infrastructure visible.",
        "source": "Sentinel-2 or Landsat",
    },
    {
        "scenario": "Agricultural Land",
        "description": "Irrigated farmland with field patterns",
        "modality": "optical",
        "suggested_query": "Are there agricultural fields? Describe the vegetation patterns.",
        "source": "Sentinel-2 (use B4/B3/B2 RGB + NDVI)",
    },
    {
        "scenario": "River/Water Body",
        "description": "Large river or reservoir",
        "modality": "optical",
        "suggested_query": "Is there a water body visible? Describe its extent.",
        "source": "Sentinel-2 or Landsat",
    },
    {
        "scenario": "Before/After Flood",
        "description": "Same area before and after flood event",
        "modality": "optical (bitemporal)",
        "suggested_query": "What changed between the two images? Is there evidence of flooding?",
        "source": "Sentinel-2 pair (download two dates)",
    },
    {
        "scenario": "Optical + SAR Pair",
        "description": "Co-registered optical and SAR over same region",
        "modality": "optical + sar",
        "suggested_query": "What additional information does the SAR image provide about this area?",
        "source": "Sentinel-2 + Sentinel-1 (same region, similar date)",
    },
    {
        "scenario": "Deforestation",
        "description": "Forest area before and after clearing",
        "modality": "optical (bitemporal)",
        "suggested_query": "Has the forest cover changed? Describe any deforestation visible.",
        "source": "Landsat or Sentinel-2 time series",
    },
]

if __name__ == "__main__":
    print("=" * 60)
    print("SatQuery AI — Demo Dataset Information")
    print("=" * 60)
    print("\nThis script provides information about legal data sources.")
    print("Visit the URLs below to download data for your demonstration.\n")

    for key, ds in DATASETS.items():
        print(f"\n{'─' * 50}")
        print(f"Dataset: {ds['name']}")
        print(f"URL:     {ds['url']}")
        print(f"License: {ds['license']}")
        print(f"Format:  {ds['format']}")
        print(f"Resolution: {ds['resolution']}")
        print(f"Best for: {ds['description']}")
        print(f"Note: {ds['note']}")

    print(f"\n\n{'=' * 60}")
    print("Demo Scenarios")
    print("=" * 60)
    for s in DEMO_SCENARIOS:
        print(f"\n[{s['scenario']}]")
        print(f"  Modality: {s['modality']}")
        print(f"  Description: {s['description']}")
        print(f"  Try asking: '{s['suggested_query']}'")
        print(f"  Data source: {s['source']}")

    print("\n\nIMPORTANT: Store downloaded images locally.")
    print("DO NOT commit GeoTIFF files to the Git repository.")
    print("Add large files to .gitignore.")
