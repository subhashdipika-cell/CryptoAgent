"""Explicit manual promotion control for validated candidate policies."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from config import SETTINGS, Settings


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def approve(symbol: str, settings: Settings = SETTINGS) -> dict:
    candidate_payload = _load(settings.candidate_policy_path)
    active_payload = _load(settings.decision_policy_path)
    candidate = next(
        (row for row in candidate_payload.get("policies", []) if row["symbol"] == symbol), None
    )
    if candidate is None:
        raise ValueError(f"no candidate policy exists for {symbol}")
    if not candidate.get("enabled", False):
        raise PermissionError(f"{symbol} candidate failed validation and cannot be approved")
    approved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    promoted = {**candidate, "approved": True, "activated_at": approved_at}
    policies = [row for row in active_payload.get("policies", []) if row["symbol"] != symbol]
    policies.append(promoted)
    active_payload["policies"] = sorted(policies, key=lambda row: row["symbol"])
    active_payload.setdefault("approval_audit", []).append(
        {
            "symbol": symbol,
            "action": "MANUAL_APPROVAL",
            "approved_at": approved_at,
            "candidate_policy": promoted,
        }
    )
    _write_atomic(settings.decision_policy_path, active_payload)
    return promoted


def status(settings: Settings = SETTINGS) -> dict:
    active = _load(settings.decision_policy_path)
    candidate = (
        _load(settings.candidate_policy_path)
        if settings.candidate_policy_path.is_file() else {"policies": []}
    )
    return {"active": active.get("policies", []), "candidate": candidate.get("policies", [])}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    approval = subparsers.add_parser("approve")
    approval.add_argument("symbol")
    args = parser.parse_args()
    if args.command == "approve":
        print(json.dumps(approve(args.symbol), indent=2))
        print("Restart CryptoAgent to load the manually approved policy.")
    else:
        print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    main()
