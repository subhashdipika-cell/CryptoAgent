"""Create a Windows virtual environment and install declared dependencies.

Model weights and secrets are intentionally not downloaded or created here.
Run ``stage_chronos_model.py`` explicitly while online, then configure secrets
through process environment variables.
"""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path


def main() -> None:
    if sys.platform != "win32":
        raise SystemExit("MetaTrader5 integration requires 64-bit Windows Python")
    root = Path(__file__).resolve().parent
    environment = root / ".venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "Scripts" / "python.exe"
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(root / "requirements.txt")], check=True)
    for directory in (root / "models", root / "data", root / "logs"):
        directory.mkdir(exist_ok=True)
    print("Environment ready. Stage Chronos weights, configure MT5 DEMO variables, then run tests.")


if __name__ == "__main__":
    main()

