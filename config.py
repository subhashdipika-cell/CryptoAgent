"""Central, environment-driven configuration for the hybrid MT5 agent.

Import this module before importing Hugging Face libraries.  The offline flags
are intentionally installed at import time so model loading cannot contact the
Hub accidentally.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("CHRONOS_MODEL_PATH", BASE_DIR / "models" / "chronos-2-base"))
TTM_MODEL_DIR = Path(os.getenv("TTM_MODEL_PATH", BASE_DIR / "models" / "granite-ttm-r3"))
TIMESFM_MODEL_DIR = Path(os.getenv("TIMESFM_MODEL_PATH", BASE_DIR / "models" / "timesfm-2.5-200m-transformers"))
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "logs"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
REPORT_DIR = Path(os.getenv("REPORT_DIR", BASE_DIR / "reports"))
POLICY_DIR = Path(os.getenv("POLICY_DIR", BASE_DIR / "policies"))
DECISION_POLICY_PATH = Path(
    os.getenv("DECISION_POLICY_PATH", POLICY_DIR / "asset_decision_policy.json")
)
CANDIDATE_POLICY_PATH = Path(
    os.getenv("CANDIDATE_POLICY_PATH", REPORT_DIR / "candidate_asset_decision_policy.json")
)
REVALIDATION_STATE_PATH = Path(
    os.getenv("REVALIDATION_STATE_PATH", DATA_DIR / "revalidation_state.json")
)

# Set these before chronos/transformers is imported anywhere else.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _optional_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else None


@dataclass(frozen=True, slots=True)
class Settings:
    symbols: tuple[str, ...] = field(
        default_factory=lambda: (
            os.getenv("MT5_BTC_SYMBOL", "BTCUSD"),
            os.getenv("MT5_XAU_SYMBOL", "XAUUSD"),
        )
    )
    bar_count: int = 500
    prediction_length: int = 5
    loop_interval_seconds: float = 60.0
    model_path: Path = MODEL_DIR
    ttm_model_path: Path = TTM_MODEL_DIR
    timesfm_model_path: Path = TIMESFM_MODEL_DIR
    max_model_ram_bytes: int = 2 * 1024**3
    torch_threads: int = 3

    hf_api_key: str = field(default_factory=lambda: os.getenv("HF_API_KEY", ""))
    hf_sentiment_model: str = "burakutf/finetuned-finbert-crypto"
    hf_inference_url: str = field(
        default_factory=lambda: os.getenv(
            "HF_INFERENCE_URL",
            "https://router.huggingface.co/hf-inference/models/"
            "burakutf/finetuned-finbert-crypto",
        )
    )
    cryptopanic_api_key: str = field(default_factory=lambda: os.getenv("CRYPTOPANIC_API_KEY", ""))
    cryptopanic_url: str = "https://cryptopanic.com/api/developer/v2/posts/"
    forexlive_api_url: str = field(default_factory=lambda: os.getenv("FOREXLIVE_API_URL", ""))
    rss_feeds: tuple[str, ...] = (
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://www.forexlive.com/feed/news/",
    )
    request_timeout_seconds: float = 12.0
    sentiment_refresh_seconds: float = 300.0
    max_headlines: int = 20

    mt5_login: int | None = field(default_factory=lambda: _optional_int("MT5_LOGIN"))
    mt5_password: str = field(default_factory=lambda: os.getenv("MT5_PASSWORD", ""))
    mt5_server: str = field(default_factory=lambda: os.getenv("MT5_SERVER", ""))
    mt5_terminal_path: str = field(default_factory=lambda: os.getenv("MT5_TERMINAL_PATH", ""))
    mt5_timeout_ms: int = 20_000
    magic_number: int = field(default_factory=lambda: int(os.getenv("MT5_MAGIC", "26081301")))
    application_name: str = field(default_factory=lambda: os.getenv("MT5_APP_NAME", "CryptoAgent"))
    strategy_name: str = field(
        default_factory=lambda: os.getenv("TRADING_STRATEGY", "ChronosFinBERT")
    )
    legacy_magic_numbers: tuple[int, ...] = (84010310,)
    journal_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("TRADE_JOURNAL_PATH", DATA_DIR / "trade_journal.db")
        )
    )
    report_dir: Path = REPORT_DIR
    decision_policy_path: Path = DECISION_POLICY_PATH
    candidate_policy_path: Path = CANDIDATE_POLICY_PATH
    revalidation_state_path: Path = REVALIDATION_STATE_PATH
    history_sync_days: int = field(
        default_factory=lambda: int(os.getenv("HISTORY_SYNC_DAYS", "3650"))
    )
    trading_enabled: bool = field(default_factory=lambda: _bool("TRADING_ENABLED"))
    require_demo_account: bool = field(default_factory=lambda: _bool("REQUIRE_DEMO_ACCOUNT", True))
    max_risk_fraction: float = field(default_factory=lambda: float(os.getenv("MAX_RISK_FRACTION", "0.02")))
    max_margin_fraction: float = field(default_factory=lambda: float(os.getenv("MAX_MARGIN_FRACTION", "0.25")))
    min_leverage: int = field(default_factory=lambda: int(os.getenv("MIN_LEVERAGE", "1")))
    max_leverage: int = field(default_factory=lambda: int(os.getenv("MAX_LEVERAGE", "500")))
    atr_period: int = 14
    stop_atr_multiple: float = 1.5
    take_profit_atr_multiple: float = 3.0
    trailing_trigger_atr: float = 1.5
    trailing_distance_atr: float = 1.0
    max_deviation_points: int = 20
    signal_threshold: float = 0.62
    predictive_mode: str = field(
        default_factory=lambda: os.getenv("PREDICTIVE_MODE", "shadow").strip().lower()
    )
    validation_bars: int = field(
        default_factory=lambda: int(os.getenv("PREDICTIVE_VALIDATION_BARS", "3000"))
    )
    validation_min_folds: int = field(
        default_factory=lambda: int(os.getenv("PREDICTIVE_MIN_FOLDS", "30"))
    )
    automatic_revalidation: bool = field(
        default_factory=lambda: _bool("AUTOMATIC_REVALIDATION", True)
    )
    revalidation_new_m15_bars: int = field(
        default_factory=lambda: int(os.getenv("REVALIDATION_NEW_M15_BARS", "500"))
    )
    dry_run: bool = field(default_factory=lambda: _bool("DRY_RUN", True))

    qwen_base_url: str = field(default_factory=lambda: os.getenv("QWEN_BASE_URL", "http://127.0.0.1:1234/v1"))
    qwen_model: str = "Qwen2.5-Coder-7B-Instruct"

    def validate(self) -> None:
        if not 0 < self.max_risk_fraction <= 0.02:
            raise ValueError("MAX_RISK_FRACTION must be in (0, 0.02]")
        if not 0 < self.max_margin_fraction <= 1:
            raise ValueError("MAX_MARGIN_FRACTION must be in (0, 1]")
        if self.bar_count < 50 or self.prediction_length != 5:
            raise ValueError("BAR_COUNT must be >= 50 and prediction length must be 5")
        if self.trading_enabled and self.dry_run:
            raise ValueError("TRADING_ENABLED and DRY_RUN cannot both be true")
        if self.trading_enabled and not self.mt5_login and not self.mt5_terminal_path:
            raise ValueError(
                "MT5_LOGIN or MT5_TERMINAL_PATH is required when order routing is enabled"
            )
        self.order_comment(self.strategy_name)
        if self.history_sync_days <= 0:
            raise ValueError("HISTORY_SYNC_DAYS must be positive")
        if self.predictive_mode not in {"shadow", "calibrated", "dedicated"}:
            raise ValueError("PREDICTIVE_MODE must be shadow, calibrated, or dedicated")
        if self.validation_bars < 500 or self.validation_min_folds < 20:
            raise ValueError("predictive validation requires >=500 bars and >=20 folds")
        if self.revalidation_new_m15_bars < 500:
            raise ValueError("REVALIDATION_NEW_M15_BARS must be >=500")

    def order_comment(self, strategy_name: str) -> str:
        application = self.application_name.strip()
        strategy = strategy_name.strip()
        if not application or not strategy:
            raise ValueError("MT5_APP_NAME and TRADING_STRATEGY cannot be blank")
        comment = f"{application}|{strategy}"
        if len(comment) > 31:
            raise ValueError("MT5 application and strategy comment exceeds 31 characters")
        return comment

    @property
    def tracked_magic_numbers(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys((*self.legacy_magic_numbers, self.magic_number)))


SETTINGS = Settings()
