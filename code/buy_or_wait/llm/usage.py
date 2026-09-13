"""Thread-safe token usage accounting for the usage report (no prompts or payloads stored)."""

import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Final

from buy_or_wait.schemas.primitives import decimal_policy

PRICES_PER_MILLION: Final[dict[str, tuple[Decimal, Decimal]]] = {
    "claude-opus-5": (Decimal("5.00"), Decimal("25.00")),
    "claude-sonnet-5": (Decimal("2.00"), Decimal("10.00")),
    "claude-haiku-4-5": (Decimal("1.00"), Decimal("5.00")),
}


@dataclass(slots=True)
class ModelUsage:
    provider: str
    model: str
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def cost(self) -> Decimal:
        input_price, output_price = PRICES_PER_MILLION.get(self.model, (Decimal(0), Decimal(0)))
        with decimal_policy():
            return (
                Decimal(self.input_tokens) * input_price
                + Decimal(self.output_tokens) * output_price
            ) / Decimal(1_000_000)


@dataclass(slots=True)
class UsageTracker:
    by_model: dict[tuple[str, str], ModelUsage] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(
        self, provider: str, model: str, input_tokens: int, output_tokens: int, *, cache_hit: bool
    ) -> None:
        with self._lock:
            usage = self.by_model.setdefault((provider, model), ModelUsage(provider, model))
            if cache_hit:
                usage.cache_hits += 1
                return
            usage.calls += 1
            usage.input_tokens += input_tokens
            usage.output_tokens += output_tokens

    def snapshot(self) -> list[ModelUsage]:
        with self._lock:
            return [self.by_model[key] for key in sorted(self.by_model)]
