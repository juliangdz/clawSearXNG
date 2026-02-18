"""Stage 2 — Engine routing based on query intent."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import structlog
import yaml

from ai_search.models import EngineConfig

logger = structlog.get_logger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "engines.yaml"
_FALLBACK_CONFIG = EngineConfig(engines=["ddg", "brave"], categories=["general"])


@lru_cache(maxsize=1)
def _load_engine_config() -> dict[str, EngineConfig]:
    """Load and cache ``config/engines.yaml``.

    Returns:
        Dict mapping intent name → :class:`EngineConfig`.
    """
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        configs: dict[str, EngineConfig] = {}
        for intent, spec in raw.get("intents", {}).items():
            configs[intent] = EngineConfig(
                engines=spec.get("engines", ["ddg"]),
                categories=spec.get("categories", ["general"]),
            )
        logger.info("router.config_loaded", intents=list(configs.keys()))
        return configs
    except Exception as exc:  # noqa: BLE001
        logger.warning("router.config_load_failed", error=str(exc))
        return {}


def route(intent: str) -> EngineConfig:
    """Return the engine and category list for *intent*.

    Falls back to the *general* config, then to a hardcoded default.

    Args:
        intent: Intent string from Stage 1, e.g. ``"research"``.

    Returns:
        :class:`EngineConfig` for that intent.
    """
    configs = _load_engine_config()
    config = configs.get(intent) or configs.get("general") or _FALLBACK_CONFIG
    logger.debug("router.routed", intent=intent, engines=config.engines)
    return config
