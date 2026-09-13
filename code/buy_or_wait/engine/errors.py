"""Typed, PII-free engine errors. Messages carry codes and identifiers, never amounts."""

from typing import ClassVar


class EngineError(RuntimeError):
    code: ClassVar[str] = "engine_error"


class MissingExchangeRateError(EngineError):
    """No ``exchange_rates.csv`` row exists for the exact settlement date and direction."""

    code: ClassVar[str] = "missing_exchange_rate"


class ForecastWindowError(EngineError):
    code: ClassVar[str] = "forecast_window_invalid"
