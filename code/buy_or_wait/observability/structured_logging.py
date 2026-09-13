"""Structured JSON logging with request correlation and defence-in-depth redaction.

Guarantees:

* one JSON object per line, UTC timestamps, ``request_id``/``run_id`` from context variables;
* only allow-listed structured fields are emitted; ``Decimal``/``float`` values are always dropped,
  so money cannot reach a log through a structured field;
* free text (messages, exception text) is scrubbed of API keys, bearer tokens, currency amounts,
  and decimal or long numbers before it is written;
* exception tracebacks are never written (only the exception type and scrubbed message).
"""

import json
import logging
import re
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Final, TextIO

ROOT_LOGGER_NAME: Final[str] = "buy_or_wait"
REDACTED_SECRET: Final[str] = "[REDACTED_SECRET]"  # noqa: S105 - redaction marker, not a secret
REDACTED_AMOUNT: Final[str] = "[REDACTED_AMOUNT]"
REDACTED_NUMBER: Final[str] = "[REDACTED_NUMBER]"

_REQUEST_ID: ContextVar[str | None] = ContextVar("bow_request_id", default=None)
_RUN_ID: ContextVar[str | None] = ContextVar("bow_run_id", default=None)

ALLOWED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "agent",
        "tool",
        "phase",
        "status",
        "step",
        "attempt",
        "duration_ms",
        "reflection_used",
        "reflection_limit",
        "record_count",
        "issue_codes",
        "error_code",
        "input_digest",
        "output_digest",
        "model_provider",
        "model_name",
        "model_calls",
        "input_tokens",
        "output_tokens",
        "cache_hit",
    }
)
"""Structured keys that may appear in logs. Nothing monetary is on this list, by design."""

_RESERVED_ATTRS: Final[frozenset[str]] = frozenset(
    set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime"}
)

_SECRET_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*\S+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
_AMOUNT_PATTERN: Final = re.compile(
    r"(?i)(?:\b(?:INR|ZAR|IDR|USD|EUR|Rp|Rs\.?)\s?|[₹$€R]\s?)-?\d[\d,]*(?:\.\d+)?"
)
_DECIMAL_NUMBER: Final = re.compile(r"(?<![\w.])-?\d[\d,]*\.\d+(?![\w.])")
_GROUPED_NUMBER: Final = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{2,3})+(?![\w.,])")
_LONG_NUMBER: Final = re.compile(r"(?<![\w\-:.])\d{4,}(?![\w\-:.])")

ScalarLogValue = str | int | bool | None
LogFieldValue = ScalarLogValue | tuple[str, ...] | list[str]


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class DisallowedLogFieldError(ValueError):
    """Raised at the call site when code tries to log a non-allow-listed field."""


def scrub_text(text: str) -> str:
    """Remove secrets and financial figures from free text."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED_SECRET, text)
    text = _AMOUNT_PATTERN.sub(REDACTED_AMOUNT, text)
    text = _DECIMAL_NUMBER.sub(REDACTED_NUMBER, text)
    text = _GROUPED_NUMBER.sub(REDACTED_NUMBER, text)
    return _LONG_NUMBER.sub(REDACTED_NUMBER, text)


def _sanitize_value(value: object) -> LogFieldValue | None:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, Decimal | float):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, tuple | list) and all(isinstance(item, str) for item in value):
        return [scrub_text(str(item)) for item in value]
    return None


class RedactingJsonFormatter(logging.Formatter):
    """Render records as single-line JSON after allow-listing and scrubbing."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": scrub_text(record.getMessage()),
            "request_id": _REQUEST_ID.get(),
            "run_id": _RUN_ID.get(),
        }
        dropped = 0
        for key, value in sorted(record.__dict__.items()):
            if key in _RESERVED_ATTRS or key.startswith("_"):
                continue
            sanitized = _sanitize_value(value) if key in ALLOWED_FIELDS else None
            if sanitized is None and value is not None:
                dropped += 1
                continue
            payload[key] = sanitized
        if dropped:
            payload["redacted_fields"] = dropped
        if record.exc_info and record.exc_info[1] is not None:
            error = record.exc_info[1]
            payload["exc_type"] = type(error).__name__
            payload["exc_message"] = scrub_text(str(error))[:300]
        return json.dumps(payload, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    *,
    stream: TextIO | None = None,
    log_file: Path | None = None,
) -> logging.Logger:
    """Attach JSON handlers to the package root logger (idempotent)."""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = RedactingJsonFormatter()
    stream_handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.setLevel(level.value)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    suffix = name.removeprefix(f"{ROOT_LOGGER_NAME}.")
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{suffix}")


@contextmanager
def request_context(request_id: str, run_id: str | None = None) -> Iterator[None]:
    """Bind ``request_id`` (and optionally ``run_id``) to every log line in this context."""
    request_token: Token[str | None] = _REQUEST_ID.set(request_id)
    run_token: Token[str | None] | None = _RUN_ID.set(run_id) if run_id is not None else None
    try:
        yield
    finally:
        _REQUEST_ID.reset(request_token)
        if run_token is not None:
            _RUN_ID.reset(run_token)


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: LogLevel = LogLevel.INFO,
    fields: Mapping[str, LogFieldValue] | None = None,
) -> None:
    """Log a short event code with allow-listed fields; unknown keys fail fast."""
    extra = dict(fields or {})
    unknown = sorted(set(extra) - ALLOWED_FIELDS)
    if unknown:
        raise DisallowedLogFieldError(f"fields not allowed in logs: {unknown}")
    logger.log(logging.getLevelNamesMapping()[level.value], event, extra=extra)
