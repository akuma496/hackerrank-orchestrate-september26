"""Strict ``output.csv`` validation policy, declared before any request is executed.

Row rules (every row):
  R1  ``request_id`` appears exactly once and belongs to ``requests.csv`` (same order, no extras).
  R2  ``0 <= amount_safe_to_pay <= requested_amount`` (canonical decimal text).
  R3  ``payment_plan`` is ``none`` or ``YYYY-MM-DD:amount`` entries joined by ``|``.
  R4  Enumerated columns hold allowed values; the date is empty or ISO; changes use the grammar.
  R5  Installment plans equal the dated schedule of a supplied installment option.
  R6  Partial-payment plans have exactly two entries totalling ``requested_amount``.
  R7  The row parses into a :class:`DecisionRecord` and passes the eligibility matrix.
  R8  ``decision_explanation`` is exactly a deterministic template rendering.
File rules: exact header order, one row per request.
"""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from buy_or_wait.output.eligibility import check_eligibility
from buy_or_wait.output.templates import expected_explanations
from buy_or_wait.schemas.audit import OptionSchedule
from buy_or_wait.schemas.decision import OUTPUT_COLUMNS, DecisionRecord, PaymentPlan
from buy_or_wait.schemas.enums import AffordabilityStatus, IssueSeverity, PaymentMethod
from buy_or_wait.schemas.tools import UserFinancialContext

_DECIMAL: Final = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")
_PLAN: Final = re.compile(
    r"none|[0-9]{4}-[0-9]{2}-[0-9]{2}:[0-9]+(?:\.[0-9]{1,2})?"
    r"(?:\|[0-9]{4}-[0-9]{2}-[0-9]{2}:[0-9]+(?:\.[0-9]{1,2})?)*"
)
_DATE: Final = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_CHANGES: Final = re.compile(
    r"none|(?:stop:event_[0-9]+|reduce_to:event_[0-9]+:[0-9]+(?:\.[0-9]{1,2})?)"
    r"(?:\|(?:stop:event_[0-9]+|reduce_to:event_[0-9]+:[0-9]+(?:\.[0-9]{1,2})?)){0,2}"
)
_STATUSES: Final = frozenset(status.value for status in AffordabilityStatus)
_METHODS: Final = frozenset(method.value for method in PaymentMethod)


class OutputValidationError(RuntimeError):
    def __init__(self, violations: Sequence["RowViolation"]) -> None:
        self.violations = tuple(violations)
        codes = sorted({violation.rule for violation in violations})
        super().__init__(f"output.csv failed validation: {len(violations)} violations {codes}")


@dataclass(frozen=True, slots=True)
class RowViolation:
    request_id: str
    rule: str
    detail: str


def validate_row(
    row: Mapping[str, str],
    context: UserFinancialContext,
    schedules: tuple[OptionSchedule, ...],
    horizon_days: int,
) -> list[RowViolation]:
    """Rules R2-R8 for one row (R1 needs the whole file)."""
    request = context.request
    rid = request.request_id
    found: list[RowViolation] = []

    def fail(rule: str, detail: str) -> None:
        found.append(RowViolation(rid, rule, detail))

    if tuple(row) != OUTPUT_COLUMNS:
        fail("R0", "row columns differ from the required header")
        return found
    if row["request_id"] != rid:
        fail("R1", "row answers a different request")
    amount_text = row["amount_safe_to_pay"]
    if not _DECIMAL.fullmatch(amount_text):
        fail("R2", "amount_safe_to_pay is not a canonical non-negative decimal")
    elif not Decimal(0) <= Decimal(amount_text) <= request.requested_amount:
        fail("R2", "amount_safe_to_pay is outside [0, requested_amount]")
    if not _PLAN.fullmatch(row["payment_plan"]):
        fail("R3", "payment_plan must be 'none' or dated amounts")
    if (
        row["affordability_status"] not in _STATUSES
        or row["recommended_payment_method"] not in _METHODS
    ):
        fail("R4", "status or method is not an allowed value")
    earliest = row["earliest_date_for_full_payment"]
    if earliest and not _DATE.fullmatch(earliest):
        fail("R4", "earliest_date_for_full_payment must be empty or YYYY-MM-DD")
    if not _CHANGES.fullmatch(row["spending_changes_needed"]):
        fail("R4", "spending_changes_needed must be 'none' or up to three changes")
    if found:
        return found

    try:
        decision = DecisionRecord.from_csv_row(row)
    except ValueError:
        fail("R7", "row violates the decision contract")
        return found

    method = decision.recommended_payment_method
    if method is PaymentMethod.INSTALLMENTS:
        installment_ids = {
            option.payment_option_id
            for option in context.payment_options
            if option.payment_method is PaymentMethod.INSTALLMENTS
        }
        if not any(
            schedule.plan == decision.payment_plan and schedule.payment_option_id in installment_ids
            for schedule in schedules
        ):
            fail("R5", "installment plan does not match a supplied option exactly")
    if method is PaymentMethod.PARTIAL_PAYMENT and (
        decision.payment_plan.payment_count != 2
        or decision.payment_plan.total() != request.requested_amount
    ):
        fail("R6", "partial payment needs two entries totalling requested_amount")
    blocking = [
        issue
        for issue in check_eligibility(decision, context, schedules, horizon_days)
        if issue.severity is IssueSeverity.ERROR
    ]
    if blocking:
        fail("R7", "; ".join(sorted({issue.detail for issue in blocking}))[:200])
    if row["decision_explanation"] not in expected_explanations(decision, context, horizon_days):
        fail("R8", "decision_explanation is not a deterministic template rendering")
    return found


def validate_file(
    header: Sequence[str], rows: Sequence[Mapping[str, str]], expected_request_ids: Sequence[str]
) -> list[RowViolation]:
    """File rules plus R1 (presence exactly once, same set and order as ``requests.csv``)."""
    violations: list[RowViolation] = []
    if tuple(header) != OUTPUT_COLUMNS:
        violations.append(RowViolation("*", "R0", "header differs from the required columns"))
    counts = Counter(row.get("request_id", "") for row in rows)
    for request_id, count in sorted(counts.items()):
        if count != 1:
            violations.append(RowViolation(request_id, "R1", "request_id must appear exactly once"))
        if request_id not in set(expected_request_ids):
            violations.append(RowViolation(request_id, "R1", "request_id is not in requests.csv"))
    for request_id in expected_request_ids:
        if request_id not in counts:
            violations.append(RowViolation(request_id, "R1", "request is missing from output"))
    if [row.get("request_id") for row in rows] != list(expected_request_ids) and not violations:
        violations.append(RowViolation("*", "R1", "rows must follow requests.csv order"))
    return violations


def plan_text_is_valid(text: str) -> bool:
    if not _PLAN.fullmatch(text):
        return False
    try:
        PaymentPlan.from_field(text)
    except ValueError:
        return False
    return True
