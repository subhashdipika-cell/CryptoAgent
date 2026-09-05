"""Explicit one-time online staging for optional foundation-model challengers."""

from __future__ import annotations

import argparse
from pathlib import Path


MODELS = {
    "ttm": ("ibm-granite/granite-timeseries-ttm-r3", "granite-ttm-r3"),
    "timesfm": ("google/timesfm-2.5-200m-transformers", "timesfm-2.5-200m-transformers"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", choices=(*MODELS, "all"))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    root = Path(__file__).resolve().parent / "models"
    selected = MODELS if args.candidate == "all" else {args.candidate: MODELS[args.candidate]}
    for _, (model_id, folder) in selected.items():
        destination = root / folder
        snapshot_download(repo_id=model_id, local_dir=destination)
        print(f"Staged {model_id} at {destination}")


if __name__ == "__main__":
    main()
