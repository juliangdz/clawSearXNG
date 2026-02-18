"""Structlog configuration — JSON in production, pretty console in development."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(environment: str, log_level: str = "INFO") -> None:
    """Configure structlog for the given environment.

    In *production* mode, emits newline-delimited JSON suitable for log
    aggregation systems.  In *development* mode, uses a colourised console
    renderer for easy reading.

    Args:
        environment: One of ``"production"`` or ``"development"`` (default).
        log_level:   Standard Python log-level name, e.g. ``"INFO"``.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if environment == "production":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level)

    # Suppress noisy third-party loggers in production.
    if environment == "production":
        for noisy in ("httpx", "httpcore", "sentence_transformers", "transformers"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
