"""Batch execution: run every request through the agent graph, gate each row, write the CSV.

Each request is isolated. Any exception, ineligible outcome, template mismatch, or row
validation failure is replaced by a deterministic fallback row, which is itself validated. The
file is validated as a whole before it is written atomically.
"""

import asyncio
import csv
import io
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from buy_or_wait.agents.fallback import FallbackReason, fallback_decision
from buy_or_wait.agents.graph import AffordabilityGraph
from buy_or_wait.data.loader import DatasetRepository
from buy_or_wait.engine.plans import build_payment_schedule
from buy_or_wait.observability.structured_logging import (
    LogLevel,
    get_logger,
    log_event,
    request_context,
)
from buy_or_wait.output.validation import OutputValidationError, validate_file, validate_row
from buy_or_wait.schemas.audit import OptionSchedule
from buy_or_wait.schemas.decision import OUTPUT_COLUMNS
from buy_or_wait.schemas.enums import PaymentMethod
from buy_or_wait.schemas.tools import PaymentScheduleInput, UserFinancialContext

_LOG = get_logger("output.batch")
DEFAULT_CONCURRENCY: Final[int] = 8


@dataclass(slots=True)
class BatchResult:
    rows: list[dict[str, str]]
    fallback_reasons: Counter[str] = field(default_factory=Counter)
    proposals: int = 0
    duration_ms: int = 0

    @property
    def fallback_count(self) -> int:
        return sum(self.fallback_reasons.values())


def installment_schedules(context: UserFinancialContext) -> tuple[OptionSchedule, ...]:
    return tuple(
        build_payment_schedule(
            PaymentScheduleInput(request_id=context.request_id, option=option)
        ).schedule
        for option in context.payment_options
        if option.payment_method is PaymentMethod.INSTALLMENTS
    )


class BatchRunner:
    def __init__(
        self,
        repository: DatasetRepository,
        graph: AffordabilityGraph,
        horizon_days: int,
        concurrency: int = DEFAULT_CONCURRENCY,
    ) -> None:
        self._repository = repository
        self._graph = graph
        self._horizon_days = horizon_days
        self._semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run(self) -> BatchResult:
        started = time.perf_counter_ns()
        outcomes = await asyncio.gather(
            *(self._one(request_id) for request_id in self._repository.request_order)
        )
        result = BatchResult(rows=[row for row, _, _ in outcomes])
        for _, reason, proposals in outcomes:
            result.proposals += proposals
            if reason is not None:
                result.fallback_reasons[reason] += 1
        violations = validate_file(OUTPUT_COLUMNS, result.rows, self._repository.request_order)
        if violations:
            raise OutputValidationError(violations)
        result.duration_ms = (time.perf_counter_ns() - started) // 1_000_000
        return result

    async def _one(self, request_id: str) -> tuple[dict[str, str], str | None, int]:
        async with self._semaphore:
            with request_context(request_id):
                context: UserFinancialContext | None = None
                try:
                    context = self._repository.context_for(request_id)
                    final, _ = await self._graph.run(context)
                    schedules = installment_schedules(context)
                    problems = validate_row(final.csv_row, context, schedules, self._horizon_days)
                    if not problems:
                        reason = (
                            FallbackReason.VERIFICATION_LIMIT.value if final.fallback_used else None
                        )
                        return final.csv_row, reason, final.proposals
                    log_event(
                        _LOG,
                        "row.rejected",
                        level=LogLevel.WARNING,
                        fields={
                            "issue_codes": sorted({problem.rule for problem in problems}),
                        },
                    )
                    return self._fallback_row(context, schedules), "row_validation", final.proposals
                except Exception as error:
                    log_event(
                        _LOG,
                        "request.failed",
                        level=LogLevel.ERROR,
                        fields={
                            "error_code": type(error).__name__.lower()[:60],
                        },
                    )
                    if context is None:
                        raise
                    return (
                        self._fallback_row(context, installment_schedules(context)),
                        "exception",
                        0,
                    )

    def _fallback_row(
        self, context: UserFinancialContext, schedules: tuple[OptionSchedule, ...]
    ) -> dict[str, str]:
        row = fallback_decision(context, FallbackReason.PROCESSING_ERROR).to_csv_row()
        problems = validate_row(row, context, schedules, self._horizon_days)
        if problems:
            raise OutputValidationError(problems)
        return row


def render_csv(rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(OUTPUT_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_output(rows: list[dict[str, str]], destination: Path) -> None:
    """Atomic write: the target is replaced only after the full file is on disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(render_csv(rows), encoding="utf-8", newline="")
    os.replace(temporary, destination)


def reread_and_validate(destination: Path, expected_request_ids: tuple[str, ...]) -> None:
    with destination.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = reader.fieldnames or []
    violations = validate_file(header, rows, expected_request_ids)
    if violations:
        raise OutputValidationError(violations)
