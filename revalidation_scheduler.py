"""Persistent completed-bar counter and isolated candidate revalidation runner."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from config import BASE_DIR, Settings


LOGGER = logging.getLogger("revalidation_scheduler")


class RevalidationScheduler:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = Path(settings.revalidation_state_path)
        self.state = self._load()
        self._task: asyncio.Task[None] | None = None
        self._process: asyncio.subprocess.Process | None = None

    def _load(self) -> dict:
        if not self.path.is_file():
            return {"symbols": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return payload if isinstance(payload.get("symbols"), dict) else {"symbols": {}}
        except (OSError, ValueError, TypeError):
            LOGGER.exception("invalid revalidation state; starting a new counter")
            return {"symbols": {}}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def observe(self, completed_m15: dict[str, tuple[int, ...]]) -> None:
        changed = False
        for symbol, timestamps in completed_m15.items():
            if not timestamps:
                continue
            timestamp = max(timestamps)
            is_new_symbol = symbol not in self.state["symbols"]
            row = self.state["symbols"].setdefault(
                symbol, {"last_m15_time": timestamp, "new_completed_bars": 0, "last_attempt_time": 0}
            )
            if is_new_symbol:
                changed = True
            previous = int(row.get("last_m15_time", timestamp))
            new_timestamps = {int(value) for value in timestamps if int(value) > previous}
            if new_timestamps:
                row["new_completed_bars"] = int(row.get("new_completed_bars", 0)) + len(new_timestamps)
                row["last_m15_time"] = timestamp
                changed = True
                LOGGER.info(
                    "%s revalidation progress %d/%d completed M15 bars",
                    symbol, row["new_completed_bars"], self.settings.revalidation_new_m15_bars,
                )
        if changed:
            self._save()
        if self.settings.automatic_revalidation and self._task is None:
            due = any(
                int(row.get("new_completed_bars", 0)) >= self.settings.revalidation_new_m15_bars
                and int(row.get("last_m15_time", 0)) > int(row.get("last_attempt_time", 0))
                for row in self.state["symbols"].values()
            )
            if due:
                for row in self.state["symbols"].values():
                    row["last_attempt_time"] = int(row.get("last_m15_time", 0))
                self._save()
                self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        LOGGER.warning("500-bar threshold reached; starting isolated candidate revalidation")
        environment = os.environ.copy()
        environment.update(
            {
                "TRADING_ENABLED": "false",
                "DRY_RUN": "true",
                "REQUIRE_DEMO_ACCOUNT": "true",
            }
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(BASE_DIR / "predictive_validation.py"),
                "--bars",
                str(self.settings.validation_bars),
                cwd=str(BASE_DIR),
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=creationflags,
            )
            stdout, stderr = await self._process.communicate()
            if self._process.returncode == 0:
                for row in self.state["symbols"].values():
                    row["new_completed_bars"] = 0
                self._save()
                LOGGER.warning(
                    "candidate revalidation complete; manual approval required: %s output=%s",
                    self.settings.candidate_policy_path,
                    stdout.decode(errors="replace").strip(),
                )
            else:
                LOGGER.error(
                    "candidate revalidation failed exit=%s error=%s",
                    self._process.returncode,
                    stderr.decode(errors="replace").strip(),
                )
        except asyncio.CancelledError:
            if self._process is not None and self._process.returncode is None:
                self._process.terminate()
                await self._process.wait()
            raise
        except Exception:
            LOGGER.exception("unable to run candidate revalidation")
        finally:
            self._process = None
            self._task = None

    async def close(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
