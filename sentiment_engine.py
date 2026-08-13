"""Asynchronous news collection and resilient serverless FinBERT scoring."""

from __future__ import annotations

import asyncio
import html
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

from config import Settings


LOGGER = logging.getLogger(__name__)
TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True, slots=True)
class SentimentResult:
    score: float
    headline_count: int
    degraded: bool = False


def _clean(value: str) -> str:
    return " ".join(html.unescape(TAG_RE.sub(" ", value or "")).split())


def parse_rss(payload: str) -> list[str]:
    root = ET.fromstring(payload)
    titles: list[str] = []
    for item in root.findall(".//item"):
        title = item.findtext("title") or ""
        description = item.findtext("description") or ""
        text = _clean(f"{title}. {description}")
        if text:
            titles.append(text)
    return titles


def normalize_sentiment(payload: Any) -> float:
    """Map common HF label payloads to [0 bearish, .5 neutral, 1 bullish]."""
    rows = payload
    if isinstance(rows, dict) and "error" in rows:
        raise ValueError(str(rows["error"]))
    if isinstance(rows, list) and rows and isinstance(rows[0], list):
        rows = rows[0]
    if not isinstance(rows, list):
        raise ValueError("unexpected sentiment response shape")
    weighted = total = 0.0
    mapping = {"bearish": 0.0, "negative": 0.0, "neutral": 0.5, "bullish": 1.0, "positive": 1.0}
    for row in rows:
        label = str(row.get("label", "")).lower()
        score = float(row.get("score", 0.0))
        value = next((number for name, number in mapping.items() if name in label), None)
        if value is not None and score >= 0:
            weighted += value * score
            total += score
    if total <= 0:
        raise ValueError("response contained no recognized labels")
    return min(1.0, max(0.0, weighted / total))


class SentimentEngine:
    def __init__(self, settings: Settings):
        try:
            import aiohttp
        except ModuleNotFoundError as error:
            raise RuntimeError("aiohttp is required; install requirements.txt") from error
        self._aiohttp = aiohttp
        self.settings = settings
        timeout = aiohttp.ClientTimeout(total=settings.request_timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=8))

    async def __aenter__(self) -> "SentimentEngine":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        if not self._session.closed:
            await self._session.close()

    async def _get_text(self, url: str, **kwargs: Any) -> str:
        async with self._session.get(url, **kwargs) as response:
            response.raise_for_status()
            return await response.text()

    async def collect_headlines(self) -> list[str]:
        jobs = [self._get_text(url) for url in self.settings.rss_feeds]
        results = await asyncio.gather(*jobs, return_exceptions=True)
        headlines: list[str] = []
        for result in results:
            if isinstance(result, Exception):
                LOGGER.warning("News feed unavailable: %s", result)
                continue
            try:
                headlines.extend(parse_rss(result))
            except (ET.ParseError, ValueError) as error:
                LOGGER.warning("Invalid RSS response: %s", error)

        if self.settings.cryptopanic_api_key:
            try:
                text = await self._get_text(
                    self.settings.cryptopanic_url,
                    params={"auth_token": self.settings.cryptopanic_api_key, "public": "true"},
                )
                import json

                headlines.extend(
                    _clean(item.get("title", "")) for item in json.loads(text).get("results", [])
                )
            except Exception as error:  # isolated optional provider
                LOGGER.warning("CryptoPanic unavailable: %s", error)
        if self.settings.forexlive_api_url:
            try:
                text = await self._get_text(self.settings.forexlive_api_url)
                import json

                rows = json.loads(text)
                if isinstance(rows, dict):
                    rows = rows.get("results", rows.get("data", []))
                headlines.extend(_clean(row.get("title", "")) for row in rows if isinstance(row, dict))
            except Exception as error:
                LOGGER.warning("ForexLive API unavailable: %s", error)
        return list(dict.fromkeys(filter(None, headlines)))[: self.settings.max_headlines]

    async def score(self) -> SentimentResult:
        headlines = await self.collect_headlines()
        if not headlines or not self.settings.hf_api_key:
            return SentimentResult(0.5, len(headlines), degraded=True)
        headers = {"Authorization": f"Bearer {self.settings.hf_api_key}"}
        try:
            async with self._session.post(
                self.settings.hf_inference_url,
                headers=headers,
                json={"inputs": headlines, "options": {"wait_for_model": False}},
            ) as response:
                if response.status in {429, 503}:
                    LOGGER.warning("HF sentiment degraded with HTTP %s", response.status)
                    return SentimentResult(0.5, len(headlines), degraded=True)
                response.raise_for_status()
                payload = await response.json()
            if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                item_scores = [normalize_sentiment(payload)]
            else:
                item_scores = [normalize_sentiment(item) for item in payload]
            return SentimentResult(sum(item_scores) / len(item_scores), len(headlines))
        except (self._aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError) as error:
            LOGGER.warning("HF sentiment fallback to neutral: %s", error)
            return SentimentResult(0.5, len(headlines), degraded=True)
