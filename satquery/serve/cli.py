"""
satquery/serve/cli.py

Command-line demo, mirroring GeoChat's `geochat/serve/cli.py`. Lets you
query one or two local images without spinning up the Streamlit app —
useful for quick checks and for the evaluation batch scripts in
`satquery/eval/`.

Usage:
    python -m satquery.serve.cli --image scene.png --query "Describe this image."
    python -m satquery.serve.cli --image-a t1.tif --image-b t2.tif \
        --query "What changed between these two dates?"
"""
from __future__ import annotations

import argparse
import json
import sys

from satquery.controller.agent_controller import AgentController
from satquery.utils.image_io import load_image


def main():
    parser = argparse.ArgumentParser(description="SatQuery AI command-line demo")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--image-a", type=str, help="Path to the earlier-date / optical image")
    parser.add_argument("--image-b", type=str, help="Path to the later-date / SAR image")
    parser.add_argument("--query", type=str, required=True, help="Natural-language query")
    args = parser.parse_args()

    if args.image:
        images = [load_image(args.image)]
    elif args.image_a and args.image_b:
        images = [load_image(args.image_a), load_image(args.image_b)]
    else:
        print("Provide either --image, or both --image-a and --image-b.", file=sys.stderr)
        sys.exit(1)

    controller = AgentController()
    result = controller.handle_query(args.query, images)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
