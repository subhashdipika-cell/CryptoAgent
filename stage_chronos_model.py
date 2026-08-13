"""One-time online staging utility; never imported by the trading runtime."""

from pathlib import Path


def main() -> None:
    from huggingface_hub import snapshot_download

    destination = Path(__file__).resolve().parent / "models" / "chronos-2-base"
    snapshot_download(repo_id="amazon/chronos-2", local_dir=destination)
    print(f"Staged amazon/chronos-2 at {destination}")


if __name__ == "__main__":
    main()

