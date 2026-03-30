from __future__ import annotations

import argparse
import json
from pathlib import Path

from drscreen.infer.service import InferenceSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fundus DR AI single-image inference.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--image", required=True, help="Path to fundus image.")
    parser.add_argument(
        "--checkpoint",
        help="Optional checkpoint override. Defaults to config.infer.checkpoint_path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    image_path = Path(args.image).resolve()
    session = InferenceSession.from_config_path(config_path, checkpoint_path=args.checkpoint)
    prediction = session.predict_image_path(image_path)
    print(json.dumps(prediction.payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
