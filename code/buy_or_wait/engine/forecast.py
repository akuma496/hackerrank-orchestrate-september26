"""Cash-flow projection and the 90-day daily balance trajectory.

Intraday convention (``DEBITS_FIRST``): on each day scheduled debits leave before that day's
credits arrive, while a recommended payment is made *after* the day's credits. So a payment on
payday may use that day's salary, but a bill due on payday cannot. The day's ``intraday_low`` is
``min(opening - scheduled_debits, closing)``.

A lump sum ``X`` paid on day ``i`` is safe when ``closing[i] - X`` and every later
``intraday_low - X`` stay at or above the minimum balance. So the payment capacity of day ``i``
is ``min(closing[i], min(intraday_low[i+1:]))`` and:

* ``amount_safe_to_pay = clamp(capacity[0] - minimum_balance, 0, requested_amount)``
* ``earliest_date_for_full_payment`` = first ``i`` with ``capacity[i] - minimum >= requested``.
"""

import calendar
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from itertools import accumulate
from typing import Final

from buy_or_wait.schemas.decision import SpendingChange
from buy_or_wait.schemas.enums import (
    Category,
    Direction,
    EventStatus,
    EvidenceKind,
    FlowOrigin,
    IntradayOrdering,
    SpendingAction,
)
from buy_or_wait.schemas.evidence import ResolvedEvidence
from buy_or_wait.schemas.primitives import decimal_policy, id_ordinal, quantize_money
from buy_or_wait.schemas.tools import (
    DailyBalance,
    EarliestFullPaymentInput,
    EarliestFullPaymentOutput,
    ForecastBalancesInput,
    ForecastBalancesOutput,
    ForecastContext,
    LedgerEntry,
    ProjectCashFlowsInput,
    ProjectCashFlowsOutput,
    ProjectedCashFlow,
    RecurringSeries,
    SafeAmountInput,
    SafeAmountOutput,
)

SCHEDULED_OVERLAP_DAYS: Final[int] = 7
"""A pending/scheduled debit replaces a projected occurrence of its category within this range."""
EXPENSE_CHANGE_CATEGORIES: Final[frozenset[Category]] = frozenset({Category.RENT, Category.HOUSING})
SALARY_OVERRIDE_KINDS: Final[frozenset[EvidenceKind]] = frozenset(
    {EvidenceKind.BASE_SALARY_CONFIRMED, EvidenceKind.INCOME_SOURCE_REDUCED}
)


def add_months(origin: date, months: int, day: int) -> date:
    year, month_index = divmod(origin.month - 1 + months, 12)
    year += origin.year
    month = month_index + 1
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def monthly_dates(last_seen: date, anchor_day: int, start: date, end: date) -> list[date]:
    dates: list[date] = []
    offset = 1
    while (candidate := add_months(last_seen, offset, anchor_day)) <= end:
        if candidate >= start:
            dates.append(candidate)
        offset += 1
    return dates


@dataclass(frozen=True, slots=True)
class _Flow:
    flow_date: date
    direction: Direction
    amount: Decimal
    origin: FlowOrigin
    category: Category | None
    source_event_id: str | None
    payment_option_id: str | None
    is_essential: bool
    source_key: str

    def order(self) -> tuple[date, int, str, str, Decimal]:
        return (
            self.flow_date,
            0 if self.direction is Direction.DEBIT else 1,
            self.origin.value,
            self.source_key,
            self.amount,
        )


# ---------------------------------------------------------------------------
# salary stream (income adjustments come only from resolved evidence)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _Payday:
    pay_date: date
    amount: Decimal
    confirmed: bool


def _evidence(context: ForecastContext, kind: EvidenceKind) -> list[ResolvedEvidence]:
    return sorted(
        (
            item
            for item in context.resolved_evidence
            if item.kind is kind
            and (item.currency is None or item.currency is context.home_currency)
        ),
        key=lambda item: (item.effective_date or date.min, item.claim_id),
    )


def _salary_stream(
    context: ForecastContext, future: Sequence[LedgerEntry], series: Sequence[RecurringSeries]
) -> tuple[list[_Payday], frozenset[str]]:
    """Primary salary paydays in the window, plus the salary series keys it supersedes."""
    start, end = context.start_date, context.end_date
    scheduled = [
        entry
        for entry in future
        if entry.direction is Direction.CREDIT
        and entry.status is EventStatus.SCHEDULED
        and entry.category is Category.SALARY
    ]
    salary_series = [
        item
        for item in series
        if item.direction is Direction.CREDIT and item.category is Category.SALARY
    ]
    paydays: list[_Payday] = []
    superseded: set[str] = set()
    if scheduled:
        first = scheduled[0]
        paydays.append(_Payday(max(first.cash_date, start), first.amount, confirmed=True))
        offset = 1
        while (following := add_months(first.cash_date, offset, first.cash_date.day)) <= end:
            paydays.append(_Payday(following, first.amount, confirmed=False))
            offset += 1
        superseded.update(
            item.series_key
            for item in salary_series
            if item.anchor_day_of_month == first.cash_date.day
            or item.projected_amount == first.amount
        )
    elif salary_series:
        primary = max(salary_series, key=lambda item: (item.projected_amount, item.series_key))
        superseded.add(primary.series_key)
        anchor = primary.anchor_day_of_month or primary.last_seen.day
        paydays.extend(
            _Payday(pay_date, primary.projected_amount, confirmed=False)
            for pay_date in monthly_dates(primary.last_seen, anchor, start, end)
        )

    for item in _evidence(context, EvidenceKind.FIRST_SALARY_CONFIRMED):
        if item.amount is None or item.effective_date is None:
            continue
        pay_date = item.effective_date
        if not paydays:
            paydays.append(_Payday(pay_date, item.amount, confirmed=True))
            offset = 1
            while (following := add_months(pay_date, offset, pay_date.day)) <= end:
                paydays.append(_Payday(following, item.amount, confirmed=False))
                offset += 1
    for item in _evidence(context, EvidenceKind.SALARY_DATE_CHANGE):
        if item.effective_date is None or not paydays:
            continue
        moved = item.effective_date
        rebuilt = [_Payday(moved, paydays[0].amount, paydays[0].confirmed)]
        offset = 1
        while (following := add_months(moved, offset, moved.day)) <= end:
            rebuilt.append(_Payday(following, paydays[0].amount, confirmed=False))
            offset += 1
        paydays = rebuilt
    for item in _evidence(context, EvidenceKind.SALARY_CHANGE):
        if item.amount is None:
            continue
        for payday in paydays:
            if item.effective_date is None or payday.pay_date >= item.effective_date:
                payday.amount = item.amount
    for kind in sorted(SALARY_OVERRIDE_KINDS, key=lambda value: value.value):
        for item in _evidence(context, kind):
            if item.amount is not None:
                for payday in paydays:
                    payday.amount = item.amount
    for item in _evidence(context, EvidenceKind.TEMPORARY_SALARY_CHANGE):
        if item.amount is not None and paydays:
            paydays[0].amount = item.amount
    if _evidence(context, EvidenceKind.INCOME_ENDED):
        paydays = [payday for payday in paydays if payday.confirmed]
    return [p for p in paydays if start <= p.pay_date <= end], frozenset(superseded)


# ---------------------------------------------------------------------------
# project_cash_flows
# ---------------------------------------------------------------------------
def project_cash_flows(payload: ProjectCashFlowsInput) -> ProjectCashFlowsOutput:
    context = payload.context
    start, end = context.start_date, context.end_date
    changes: dict[str, SpendingChange] = {
        change.event_id: change for change in payload.spending_changes
    }
    future = sorted(
        (
            entry
            for entry in context.ledger.entries
            if entry.status in {EventStatus.PENDING, EventStatus.SCHEDULED}
        ),
        key=lambda entry: (entry.cash_date, id_ordinal(entry.event_id)),
    )
    series = sorted(context.recurrence.recurring, key=lambda item: item.series_key)
    flows: list[_Flow] = []

    scheduled_debits: list[LedgerEntry] = []
    for entry in future:
        flow_date = max(entry.cash_date, start)
        if flow_date > end or entry.amount <= 0:
            continue
        if entry.direction is Direction.DEBIT:
            scheduled_debits.append(entry)
            origin = (
                FlowOrigin.PENDING_DEBIT
                if entry.status is EventStatus.PENDING
                else FlowOrigin.SCHEDULED_EVENT
            )
            flows.append(
                _Flow(
                    flow_date,
                    Direction.DEBIT,
                    entry.amount,
                    origin,
                    entry.category,
                    entry.event_id,
                    None,
                    True,
                    entry.event_id,
                )
            )
        elif entry.category is not Category.SALARY:
            flows.append(
                _Flow(
                    flow_date,
                    Direction.CREDIT,
                    entry.amount,
                    FlowOrigin.CONFIRMED_INCOME,
                    entry.category,
                    entry.event_id,
                    None,
                    False,
                    entry.event_id,
                )
            )

    paydays, superseded = _salary_stream(context, future, series)
    flows.extend(
        _Flow(
            payday.pay_date,
            Direction.CREDIT,
            payday.amount,
            FlowOrigin.CONFIRMED_INCOME,
            Category.SALARY,
            None,
            None,
            False,
            f"salary:{index:03d}",
        )
        for index, payday in enumerate(paydays)
    )

    rent_multiplier = Decimal(1)
    for item in _evidence(context, EvidenceKind.RECURRING_EXPENSE_CHANGE):
        if item.percentage is not None:
            with decimal_policy():
                rent_multiplier *= Decimal(1) + item.percentage / Decimal(100)

    for recurring_item in series:
        if recurring_item.series_key in superseded or recurring_item.anchor_day_of_month is None:
            continue
        amount = recurring_item.projected_amount
        change = changes.get(recurring_item.latest_event_id)
        if change is not None and change.action is SpendingAction.STOP:
            continue
        if change is not None and change.new_amount is not None:
            amount = change.new_amount
        if (
            recurring_item.direction is Direction.DEBIT
            and recurring_item.category in EXPENSE_CHANGE_CATEGORIES
        ):
            with decimal_policy():
                amount = quantize_money(amount * rent_multiplier)
        for flow_date in monthly_dates(
            recurring_item.last_seen, recurring_item.anchor_day_of_month, start, end
        ):
            if recurring_item.direction is Direction.DEBIT and any(
                entry.category is recurring_item.category
                and abs((entry.cash_date - flow_date).days) <= SCHEDULED_OVERLAP_DAYS
                for entry in scheduled_debits
            ):
                continue
            flows.append(
                _Flow(
                    flow_date,
                    recurring_item.direction,
                    amount,
                    FlowOrigin.RECURRING_SERIES,
                    recurring_item.category,
                    recurring_item.latest_event_id,
                    None,
                    recurring_item.is_essential,
                    recurring_item.series_key,
                )
            )

    for adjustment in _evidence(context, EvidenceKind.ONE_TIME_INCOME_ADJUSTMENT):
        pay_date = adjustment.effective_date or (paydays[0].pay_date if paydays else None)
        if adjustment.amount and pay_date is not None and start <= pay_date <= end:
            flows.append(
                _Flow(
                    pay_date,
                    Direction.CREDIT,
                    adjustment.amount,
                    FlowOrigin.EVIDENCE_ADJUSTMENT,
                    Category.SALARY,
                    None,
                    None,
                    False,
                    adjustment.claim_id,
                )
            )
    for invoice in _evidence(context, EvidenceKind.CONFIRMED_INVOICE_SETTLEMENT):
        if invoice.amount and invoice.effective_date and start <= invoice.effective_date <= end:
            flows.append(
                _Flow(
                    invoice.effective_date,
                    Direction.CREDIT,
                    invoice.amount,
                    FlowOrigin.EVIDENCE_ADJUSTMENT,
                    None,
                    None,
                    None,
                    False,
                    invoice.claim_id,
                )
            )

    for profile in sorted(context.recurrence.variable_spending, key=lambda p: p.category.value):
        if profile.projected_amount_per_period <= 0:
            continue
        step = timedelta(days=profile.period_days)
        flow_date = profile.last_seen + step
        while flow_date <= end:
            if flow_date >= start:
                flows.append(
                    _Flow(
                        flow_date,
                        Direction.DEBIT,
                        profile.projected_amount_per_period,
                        FlowOrigin.VARIABLE_SPENDING,
                        profile.category,
                        None,
                        None,
                        profile.is_essential,
                        f"variable:{profile.category.value}",
                    )
                )
            flow_date += step

    for payment in payload.plan_payments.payments:
        if start <= payment.payment_date <= end:
            flows.append(
                _Flow(
                    payment.payment_date,
                    Direction.DEBIT,
                    payment.amount,
                    FlowOrigin.PLAN_PAYMENT,
                    None,
                    None,
                    payload.payment_option_id,
                    True,
                    "plan",
                )
            )

    ordered = sorted(flows, key=_Flow.order)
    projected = tuple(
        ProjectedCashFlow(
            flow_id=f"d{flow.flow_date:%Y%m%d}-{index:05d}",
            flow_date=flow.flow_date,
            direction=flow.direction,
            amount=quantize_money(flow.amount),
            origin=flow.origin,
            category=flow.category,
            source_event_id=flow.source_event_id,
            payment_option_id=flow.payment_option_id,
            is_essential=flow.is_essential,
        )
        for index, flow in enumerate(ordered)
        if quantize_money(flow.amount) > 0
    )
    return ProjectCashFlowsOutput(
        request_id=payload.request_id, start_date=start, end_date=end, flows=projected
    )


# ---------------------------------------------------------------------------
# forecast_balances
# ---------------------------------------------------------------------------
def forecast_balances(payload: ForecastBalancesInput) -> ForecastBalancesOutput:
    start, end = payload.start_date, payload.end_date
    size = (end - start).days + 1
    zero = Decimal(0)
    credits = [zero] * size
    scheduled = [zero] * size
    planned = [zero] * size
    with decimal_policy():
        for flow in payload.flows:
            index = (flow.flow_date - start).days
            if flow.direction is Direction.CREDIT:
                credits[index] += flow.amount
            elif flow.origin is FlowOrigin.PLAN_PAYMENT:
                planned[index] += flow.amount
            else:
                scheduled[index] += flow.amount
        net = [c - s - p for c, s, p in zip(credits, scheduled, planned, strict=True)]
        closings = list(accumulate(net, initial=payload.opening_balance))[1:]
        openings = [payload.opening_balance, *closings[:-1]]
        days: list[DailyBalance] = []
        for offset in range(size):
            opening, closing = openings[offset], closings[offset]
            if payload.intraday_ordering is IntradayOrdering.DEBITS_FIRST:
                low = min(opening - scheduled[offset], closing)
            else:
                low = min(opening, closing)
            days.append(
                DailyBalance(
                    balance_date=start + timedelta(days=offset),
                    opening_balance=opening,
                    total_credits=credits[offset],
                    total_debits=scheduled[offset] + planned[offset],
                    intraday_low=low,
                    closing_balance=closing,
                )
            )
    lowest = min(days, key=lambda day: (day.intraday_low, day.balance_date))
    return ForecastBalancesOutput(
        request_id=payload.request_id,
        start_date=start,
        end_date=end,
        minimum_balance=payload.minimum_balance,
        days=tuple(days),
        lowest_balance=lowest.intraday_low,
        lowest_balance_date=lowest.balance_date,
        breaches_minimum=lowest.intraday_low < payload.minimum_balance,
    )


def forecast_context(
    context: ForecastContext, request_id: str, **projection: object
) -> tuple[ProjectCashFlowsOutput, ForecastBalancesOutput]:
    """Project flows for ``context`` (optionally with a plan / changes) and simulate balances."""
    flows = project_cash_flows(
        ProjectCashFlowsInput.model_validate(
            {"request_id": request_id, "context": context, **projection}
        )
    )
    balances = forecast_balances(
        ForecastBalancesInput(
            request_id=request_id,
            start_date=context.start_date,
            end_date=context.end_date,
            opening_balance=context.opening_balance,
            minimum_balance=context.minimum_balance,
            intraday_ordering=context.intraday_ordering,
            flows=flows.flows,
        )
    )
    return flows, balances


# ---------------------------------------------------------------------------
# capacity, safe amount, earliest date
# ---------------------------------------------------------------------------
def payment_capacities(forecast: ForecastBalancesOutput) -> list[tuple[Decimal, date]]:
    """For each day: the largest balance a same-day payment can draw on, and its binding date."""
    days = forecast.days
    suffix: list[tuple[Decimal, date] | None] = [None] * (len(days) + 1)
    for index in range(len(days) - 1, -1, -1):
        current = (days[index].intraday_low, days[index].balance_date)
        later = suffix[index + 1]
        suffix[index] = current if later is None or current[0] <= later[0] else later
    capacities: list[tuple[Decimal, date]] = []
    for index, day in enumerate(days):
        later = suffix[index + 1]
        own = (day.closing_balance, day.balance_date)
        capacities.append(own if later is None or own[0] <= later[0] else later)
    return capacities


def compute_safe_amount(payload: SafeAmountInput) -> SafeAmountOutput:
    forecast = payload.baseline_forecast
    if forecast.start_date != payload.request_date:
        raise ValueError("baseline forecast must start on request_date")
    capacity, binding = payment_capacities(forecast)[0]
    with decimal_policy():
        headroom = capacity - forecast.minimum_balance
    safe = min(max(headroom, Decimal(0)), payload.requested_amount)
    return SafeAmountOutput(
        request_id=payload.request_id,
        requested_amount=payload.requested_amount,
        minimum_headroom=headroom,
        binding_date=binding,
        amount_safe_to_pay=safe,
    )


def find_earliest_full_payment_date(payload: EarliestFullPaymentInput) -> EarliestFullPaymentOutput:
    forecast = payload.baseline_forecast
    earliest: date | None = None
    with decimal_policy():
        for (capacity, _), day in zip(payment_capacities(forecast), forecast.days, strict=True):
            if capacity - forecast.minimum_balance >= payload.requested_amount:
                earliest = day.balance_date
                break
    return EarliestFullPaymentOutput(
        request_id=payload.request_id,
        search_start=forecast.start_date,
        search_end=forecast.end_date,
        earliest_date=earliest,
    )
