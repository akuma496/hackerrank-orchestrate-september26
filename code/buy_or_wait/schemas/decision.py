"""Decision output contract: payment plans, spending changes, and the ``output.csv`` row.

Row-intrinsic rules from the problem statement are enforced as validators, so a
``DecisionRecord`` that contradicts itself (e.g. ``wait`` with no date) cannot exist. Rules that
need the request, profile, or options live in :mod:`buy_or_wait.schemas.audit`.
"""

import re
from collections.abc import Mapping
from decimal import Decimal
from itertools import pairwise
from types import MappingProxyType
from typing import Annotated, Final, Self

from pydantic import AfterValidator, field_validator, model_validator

from buy_or_wait.schemas.base import StrictModel
from buy_or_wait.schemas.enums import (
    AffordabilityStatus,
    AffordabilityStatusField,
    OutputColumn,
    PaymentMethod,
    PaymentMethodField,
    SpendingAction,
    SpendingActionField,
)
from buy_or_wait.schemas.primitives import (
    EventId,
    ExplanationText,
    IsoDate,
    NonNegativeMoney,
    OptionalIsoDate,
    OptionalPositiveMoney,
    PositiveMoney,
    RequestId,
    decimal_policy,
    format_amount,
    format_plan_amount,
    is_blank,
    parse_decimal,
    parse_iso_date,
)

NONE_TOKEN: Final[str] = "none"  # noqa: S105 - output-format sentinel, not a credential
MAX_SPENDING_CHANGES: Final[int] = 3
OUTPUT_COLUMNS: Final[tuple[str, ...]] = tuple(column.value for column in OutputColumn)

_PLAN_ENTRY: Final = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2}):([0-9]+(?:\.[0-9]+)?)")
_STOP_ENTRY: Final = re.compile(r"stop:(event_[0-9]+)")
_REDUCE_ENTRY: Final = re.compile(r"reduce_to:(event_[0-9]+):([0-9]+(?:\.[0-9]+)?)")

_STATUSES_BY_METHOD: Final[Mapping[PaymentMethod, frozenset[AffordabilityStatus]]] = (
    MappingProxyType(
        {
            PaymentMethod.FULL_PAYMENT: frozenset(
                {AffordabilityStatus.AFFORDABLE_NOW, AffordabilityStatus.AFFORDABLE_WITH_PLAN}
            ),
            PaymentMethod.PARTIAL_PAYMENT: frozenset({AffordabilityStatus.AFFORDABLE_WITH_PLAN}),
            PaymentMethod.INSTALLMENTS: frozenset({AffordabilityStatus.AFFORDABLE_WITH_PLAN}),
            PaymentMethod.WAIT: frozenset({AffordabilityStatus.AFFORDABLE_LATER}),
            PaymentMethod.NOT_RECOMMENDED: frozenset({AffordabilityStatus.NOT_AFFORDABLE}),
        }
    )
)


class ScheduledPayment(StrictModel):
    payment_date: IsoDate
    amount: PositiveMoney

    def to_field(self) -> str:
        return f"{self.payment_date.isoformat()}:{format_plan_amount(self.amount)}"


class PaymentPlan(StrictModel):
    """Chronological payments; the empty plan serialises as ``none``."""

    payments: tuple[ScheduledPayment, ...] = ()

    @field_validator("payments")
    @classmethod
    def _strictly_chronological(
        cls, value: tuple[ScheduledPayment, ...]
    ) -> tuple[ScheduledPayment, ...]:
        for earlier, later in pairwise(value):
            if later.payment_date <= earlier.payment_date:
                raise ValueError("payments must be in strictly increasing date order")
        return value

    @property
    def payment_count(self) -> int:
        return len(self.payments)

    @property
    def is_empty(self) -> bool:
        return not self.payments

    def total(self) -> Decimal:
        with decimal_policy():
            return sum((payment.amount for payment in self.payments), start=Decimal(0))

    def to_field(self) -> str:
        if not self.payments:
            return NONE_TOKEN
        return "|".join(payment.to_field() for payment in self.payments)

    @classmethod
    def from_field(cls, text: str) -> Self:
        if text == NONE_TOKEN:
            return cls()
        payments: list[ScheduledPayment] = []
        for entry in text.split("|"):
            match = _PLAN_ENTRY.fullmatch(entry)
            if match is None:
                raise ValueError("payment_plan entries must look like 'YYYY-MM-DD:amount'")
            payments.append(
                ScheduledPayment(
                    payment_date=parse_iso_date(match.group(1)),
                    amount=parse_decimal(match.group(2)),
                )
            )
        return cls(payments=tuple(payments))


class SpendingChange(StrictModel):
    """``stop:<event_id>`` or ``reduce_to:<event_id>:<new_amount>``."""

    action: SpendingActionField
    event_id: EventId
    new_amount: OptionalPositiveMoney = None

    @model_validator(mode="after")
    def _amount_matches_action(self) -> Self:
        if self.action is SpendingAction.STOP and self.new_amount is not None:
            raise ValueError("stop changes carry no new_amount")
        if self.action is SpendingAction.REDUCE_TO and self.new_amount is None:
            raise ValueError("reduce_to changes require a positive new_amount")
        return self

    def to_field(self) -> str:
        if self.new_amount is None:
            return f"{self.action.value}:{self.event_id}"
        return f"{self.action.value}:{self.event_id}:{format_plan_amount(self.new_amount)}"

    @classmethod
    def from_field(cls, text: str) -> Self:
        stop = _STOP_ENTRY.fullmatch(text)
        if stop is not None:
            return cls(action=SpendingAction.STOP, event_id=stop.group(1))
        reduce = _REDUCE_ENTRY.fullmatch(text)
        if reduce is not None:
            return cls(
                action=SpendingAction.REDUCE_TO,
                event_id=reduce.group(1),
                new_amount=parse_decimal(reduce.group(2)),
            )
        raise ValueError("spending changes must be 'stop:<event_id>' or 'reduce_to:<id>:<amount>'")


def _check_change_set(value: tuple[SpendingChange, ...]) -> tuple[SpendingChange, ...]:
    if len(value) > MAX_SPENDING_CHANGES:
        raise ValueError(f"at most {MAX_SPENDING_CHANGES} spending changes are allowed")
    event_ids = [change.event_id for change in value]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("an event can be stopped or reduced, not both, and only once")
    return value


SpendingChangeSet = Annotated[tuple[SpendingChange, ...], AfterValidator(_check_change_set)]


def format_spending_changes(changes: tuple[SpendingChange, ...]) -> str:
    return "|".join(change.to_field() for change in changes) if changes else NONE_TOKEN


def parse_spending_changes(text: str) -> tuple[SpendingChange, ...]:
    if text == NONE_TOKEN:
        return ()
    return _check_change_set(tuple(SpendingChange.from_field(entry) for entry in text.split("|")))


class DecisionRecord(StrictModel):
    """One ``output.csv`` row with its row-intrinsic invariants."""

    request_id: RequestId
    amount_safe_to_pay: NonNegativeMoney
    affordability_status: AffordabilityStatusField
    recommended_payment_method: PaymentMethodField
    payment_plan: PaymentPlan
    earliest_date_for_full_payment: OptionalIsoDate
    spending_changes_needed: SpendingChangeSet
    decision_explanation: ExplanationText

    @model_validator(mode="after")
    def _row_is_coherent(self) -> Self:
        method = self.recommended_payment_method
        status = self.affordability_status
        plan = self.payment_plan
        earliest = self.earliest_date_for_full_payment
        changes = self.spending_changes_needed

        if status not in _STATUSES_BY_METHOD[method]:
            raise ValueError("affordability_status is incompatible with the payment method")

        if method is PaymentMethod.NOT_RECOMMENDED:
            if not plan.is_empty or changes:
                raise ValueError("not_recommended has payment_plan 'none' and no spending changes")
        elif method in {PaymentMethod.FULL_PAYMENT, PaymentMethod.WAIT}:
            if plan.payment_count != 1:
                raise ValueError("full_payment and wait use exactly one payment")
        elif method is PaymentMethod.PARTIAL_PAYMENT:
            if plan.payment_count != 2:
                raise ValueError("partial_payment uses exactly two payments")
        elif plan.payment_count < 2:
            raise ValueError("installments use at least two payments")

        if status is AffordabilityStatus.AFFORDABLE_NOW:
            if earliest is None or changes:
                raise ValueError("affordable_now needs an earliest date and no spending changes")
            if plan.payments[0].payment_date != earliest:
                raise ValueError("affordable_now pays in full on request_date (= earliest date)")

        if (
            method is PaymentMethod.FULL_PAYMENT
            and status is AffordabilityStatus.AFFORDABLE_WITH_PLAN
            and not changes
        ):
            raise ValueError("full_payment is only a plan when spending changes are required")

        if method is PaymentMethod.WAIT:
            if earliest is None or plan.payments[0].payment_date != earliest:
                raise ValueError("wait pays in full on earliest_date_for_full_payment")
            if changes:
                raise ValueError("wait is judged before optional spending changes")

        if method is PaymentMethod.PARTIAL_PAYMENT:
            first, second = plan.payments
            if self.amount_safe_to_pay <= 0 or first.amount != self.amount_safe_to_pay:
                raise ValueError("partial_payment pays amount_safe_to_pay first")
            if earliest is None or second.payment_date != earliest:
                raise ValueError("partial_payment pays the remainder on the earliest full date")
        return self

    def to_csv_row(self) -> dict[str, str]:
        earliest = self.earliest_date_for_full_payment
        return {
            OutputColumn.REQUEST_ID.value: self.request_id,
            OutputColumn.AMOUNT_SAFE_TO_PAY.value: format_amount(self.amount_safe_to_pay),
            OutputColumn.AFFORDABILITY_STATUS.value: self.affordability_status.value,
            OutputColumn.RECOMMENDED_PAYMENT_METHOD.value: self.recommended_payment_method.value,
            OutputColumn.PAYMENT_PLAN.value: self.payment_plan.to_field(),
            OutputColumn.EARLIEST_DATE_FOR_FULL_PAYMENT.value: (
                "" if earliest is None else earliest.isoformat()
            ),
            OutputColumn.SPENDING_CHANGES_NEEDED.value: format_spending_changes(
                self.spending_changes_needed
            ),
            OutputColumn.DECISION_EXPLANATION.value: self.decision_explanation,
        }

    @classmethod
    def from_csv_row(cls, row: Mapping[str, str]) -> Self:
        earliest = row[OutputColumn.EARLIEST_DATE_FOR_FULL_PAYMENT.value]
        return cls(
            request_id=row[OutputColumn.REQUEST_ID.value],
            amount_safe_to_pay=parse_decimal(row[OutputColumn.AMOUNT_SAFE_TO_PAY.value]),
            affordability_status=AffordabilityStatus(row[OutputColumn.AFFORDABILITY_STATUS.value]),
            recommended_payment_method=PaymentMethod(
                row[OutputColumn.RECOMMENDED_PAYMENT_METHOD.value]
            ),
            payment_plan=PaymentPlan.from_field(row[OutputColumn.PAYMENT_PLAN.value]),
            earliest_date_for_full_payment=None if is_blank(earliest) else parse_iso_date(earliest),
            spending_changes_needed=parse_spending_changes(
                row[OutputColumn.SPENDING_CHANGES_NEEDED.value]
            ),
            decision_explanation=row[OutputColumn.DECISION_EXPLANATION.value],
        )
