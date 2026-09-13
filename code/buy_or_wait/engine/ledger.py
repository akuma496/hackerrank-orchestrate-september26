"""Currency normalisation, inclusion/exclusion rules, and recurrence detection.

Inclusion: settled cash records (history), pending debits, scheduled debits and credits.
Exclusion (each flagged with an :class:`ExclusionReason`, never silently dropped): failed and
cancelled records, pending credits (refunds, payouts, prizes), unrealized non-cash valuations,
pending charges that duplicate a settled original, and cash records whose amount is still
missing after image evidence. Foreign-currency amounts use the exact settlement-date rate from
``exchange_rates.csv`` in the stated direction; a missing rate raises
:class:`MissingExchangeRateError`.
"""

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import date
from decimal import Decimal
from itertools import pairwise
from typing import Final

from buy_or_wait.engine.errors import MissingExchangeRateError
from buy_or_wait.schemas.entities import ExchangeRate, FinancialEvent, FinancialProfile
from buy_or_wait.schemas.enums import (
    AmountBasis,
    AmountSource,
    Cadence,
    Category,
    Currency,
    Direction,
    EventStatus,
    EventType,
    EvidenceKind,
    ExclusionReason,
    Flexibility,
)
from buy_or_wait.schemas.evidence import ResolvedEvidence
from buy_or_wait.schemas.primitives import decimal_policy, id_ordinal, quantize_money
from buy_or_wait.schemas.tools import (
    CurrencyConversionInput,
    CurrencyConversionOutput,
    DetectRecurrenceInput,
    DetectRecurrenceOutput,
    LedgerEntry,
    LedgerExclusion,
    NormalizeLedgerInput,
    NormalizeLedgerOutput,
    RecurringSeries,
    VariableSpendingProfile,
)

VARIABLE_CATEGORIES: Final[frozenset[Category]] = frozenset(
    {Category.GROCERIES, Category.TRANSPORT, Category.DINING}
)
"""Irregular essential spending forecast by observed cadence (calibrated on the samples)."""

NON_RECURRING_TYPES: Final[frozenset[EventType]] = frozenset(
    {EventType.REFUND, EventType.INVESTMENT_PURCHASE, EventType.INVESTMENT_SALE}
)
NON_RECURRING_CATEGORIES: Final[frozenset[Category]] = frozenset(
    {Category.WINDFALL, Category.WORK_EXPENSE, Category.INVESTMENT}
)
MONTHLY_GAP_DAYS: Final[tuple[int, int]] = (25, 35)
MIN_MONTHLY_OCCURRENCES: Final[int] = 2
MIN_VARIABLE_OCCURRENCES: Final[int] = 3
MAX_VARIABLE_PERIOD_DAYS: Final[int] = 31

RateKey = tuple[date, Currency, Currency]


# ---------------------------------------------------------------------------
# convert_currency
# ---------------------------------------------------------------------------
def rate_index(rates: Iterable[ExchangeRate]) -> dict[RateKey, Decimal]:
    ordered = sorted(
        rates, key=lambda rate: (rate.rate_date, rate.from_currency.value, rate.to_currency.value)
    )
    return {(rate.rate_date, rate.from_currency, rate.to_currency): rate.rate for rate in ordered}


def convert_currency(
    payload: CurrencyConversionInput, rates: Mapping[RateKey, Decimal]
) -> CurrencyConversionOutput:
    if payload.source_currency is payload.target_currency:
        rate = Decimal(1)
        converted = payload.amount
    else:
        key = (payload.settlement_date, payload.source_currency, payload.target_currency)
        found = rates.get(key)
        if found is None:
            raise MissingExchangeRateError(
                f"{payload.request_id}: no {payload.source_currency.value}->"
                f"{payload.target_currency.value} rate on {payload.settlement_date.isoformat()}"
            )
        rate = found
        with decimal_policy():
            converted = quantize_money(payload.amount * rate)
    return CurrencyConversionOutput(
        request_id=payload.request_id,
        source_amount=payload.amount,
        source_currency=payload.source_currency,
        target_currency=payload.target_currency,
        rate_date=payload.settlement_date,
        rate_applied=rate,
        converted_amount=converted,
    )


# ---------------------------------------------------------------------------
# normalize_ledger
# ---------------------------------------------------------------------------
def _event_order(event: FinancialEvent) -> tuple[date, int]:
    return (event.cash_date, id_ordinal(event.event_id))


def _duplicate_pending_charges(events: Mapping[str, FinancialEvent]) -> frozenset[str]:
    """Pending debits linked to a settled debit of the same amount are duplicate records."""
    duplicates: set[str] = set()
    for event in events.values():
        linked = events.get(event.linked_event_id) if event.linked_event_id else None
        if (
            linked is not None
            and event.status is EventStatus.PENDING
            and event.direction is Direction.DEBIT
            and linked.status is EventStatus.SETTLED
            and linked.direction is Direction.DEBIT
            and linked.amount is not None
            and linked.amount == event.amount
        ):
            duplicates.add(event.event_id)
    return frozenset(duplicates)


def _exclusion_reason(event: FinancialEvent, duplicates: frozenset[str]) -> ExclusionReason | None:
    if event.direction is Direction.NON_CASH:
        return ExclusionReason.UNREALIZED_NON_CASH
    if event.status is EventStatus.FAILED:
        return ExclusionReason.FAILED
    if event.status is EventStatus.CANCELLED:
        return ExclusionReason.CANCELLED
    if event.status is EventStatus.PENDING and event.direction is Direction.CREDIT:
        return ExclusionReason.PENDING_CREDIT
    if event.event_id in duplicates:
        return ExclusionReason.DUPLICATE_RECORD
    return None


def normalize_ledger(payload: NormalizeLedgerInput) -> NormalizeLedgerOutput:
    rates = rate_index(payload.exchange_rates)
    events = sorted(payload.events, key=_event_order)
    by_id = {event.event_id: event for event in events}
    duplicates = _duplicate_pending_charges(by_id)
    document_amounts: dict[str, ResolvedEvidence] = {
        item.related_event_id: item
        for item in sorted(payload.resolved_evidence, key=lambda item: item.claim_id)
        if item.kind is EvidenceKind.DOCUMENT_AMOUNT
        and item.related_event_id is not None
        and item.amount is not None
    }

    entries: list[LedgerEntry] = []
    exclusions: list[LedgerExclusion] = []
    for event in events:
        reason = _exclusion_reason(event, duplicates)
        if reason is not None:
            exclusions.append(LedgerExclusion(event_id=event.event_id, reason=reason))
            continue
        amount, source = event.amount, AmountSource.DATASET
        if amount is None:
            evidence = document_amounts.get(event.event_id)
            if evidence is None or evidence.amount is None:
                exclusions.append(
                    LedgerExclusion(
                        event_id=event.event_id, reason=ExclusionReason.UNRESOLVED_AMOUNT
                    )
                )
                continue
            amount, source = evidence.amount, AmountSource.IMAGE_EVIDENCE
        minimum = event.minimum_allowed_amount
        if event.currency is not payload.home_currency:
            amount = _to_home(payload, rates, event, amount)
            minimum = None if minimum is None else _to_home(payload, rates, event, minimum)
            source = AmountSource.FX_CONVERSION
        entries.append(
            LedgerEntry(
                event_id=event.event_id,
                cash_date=event.cash_date,
                direction=event.direction,
                amount=amount,
                status=event.status,
                category=event.category,
                description=event.description,
                flexibility=event.flexibility,
                minimum_allowed_amount=minimum,
                amount_source=source,
            )
        )
    return NormalizeLedgerOutput(
        request_id=payload.request_id, entries=tuple(entries), exclusions=tuple(exclusions)
    )


def _to_home(
    payload: NormalizeLedgerInput,
    rates: Mapping[RateKey, Decimal],
    event: FinancialEvent,
    amount: Decimal,
) -> Decimal:
    return convert_currency(
        CurrencyConversionInput(
            request_id=payload.request_id,
            amount=amount,
            source_currency=event.currency,
            target_currency=payload.home_currency,
            settlement_date=event.cash_date,
        ),
        rates,
    ).converted_amount


# ---------------------------------------------------------------------------
# detect_recurrence
# ---------------------------------------------------------------------------
_SLUG: Final = re.compile(r"[^a-z0-9]+")


def series_key(direction: Direction, category: Category, description: str) -> str:
    slug = _SLUG.sub("_", description.lower()).strip("_") or "item"
    return f"{direction.value}:{category.value}:{slug}"[:160]


def mean_amount(amounts: Iterable[Decimal]) -> Decimal:
    values = list(amounts)
    with decimal_policy():
        return quantize_money(sum(values, start=Decimal(0)) / Decimal(len(values)))


def _modal(values: Iterable[int]) -> tuple[int, int]:
    """Most common value (ties -> smallest) and its count."""
    counts = Counter(values)
    best = min(counts.items(), key=lambda item: (-item[1], item[0]))
    return best


def _is_monthly(entries: list[LedgerEntry]) -> int | None:
    """Anchor day when the history supports a monthly cadence, else ``None``."""
    if len(entries) < MIN_MONTHLY_OCCURRENCES:
        return None
    dates = [entry.cash_date for entry in entries]
    gaps = [(later - earlier).days for earlier, later in pairwise(dates)]
    low, high = MONTHLY_GAP_DAYS
    if any(not low <= gap <= high for gap in gaps):
        return None
    anchor, count = _modal(day.day for day in dates)
    return anchor if count >= len(dates) - 1 else None


def detect_recurrence(payload: DetectRecurrenceInput) -> DetectRecurrenceOutput:
    profile: FinancialProfile = payload.profile
    protected = profile.expense_categories_to_protect
    history = sorted(
        (
            entry
            for entry in payload.ledger.entries
            if entry.status is EventStatus.SETTLED and entry.cash_date < payload.request_date
        ),
        key=lambda entry: (entry.cash_date, id_ordinal(entry.event_id)),
    )
    groups: dict[tuple[Direction, Category, str], list[LedgerEntry]] = defaultdict(list)
    for entry in history:
        if entry.category in NON_RECURRING_CATEGORIES:
            continue
        groups[(entry.direction, entry.category, entry.description)].append(entry)

    recurring: list[RecurringSeries] = []
    consumed: set[str] = set()
    for (direction, category, description), entries in sorted(
        groups.items(), key=lambda item: (item[0][0].value, item[0][1].value, item[0][2])
    ):
        anchor = _is_monthly(entries)
        latest = entries[-1]
        if anchor is None or latest.amount <= 0:
            continue
        if category in VARIABLE_CATEGORIES and latest.flexibility is Flexibility.FIXED:
            continue  # rotating descriptions inside a variable stream are not a separate series
        consumed.update(entry.event_id for entry in entries)
        is_protected = category in protected
        recurring.append(
            RecurringSeries(
                series_key=series_key(direction, category, description),
                category=category,
                description=description,
                direction=direction,
                cadence=Cadence.MONTHLY,
                anchor_day_of_month=anchor,
                projected_amount=latest.amount,
                amount_basis=AmountBasis.LATEST,
                member_event_ids=tuple(entry.event_id for entry in entries),
                latest_event_id=latest.event_id,
                first_seen=entries[0].cash_date,
                last_seen=latest.cash_date,
                flexibility=latest.flexibility
                if direction is Direction.DEBIT
                else Flexibility.FIXED,
                minimum_allowed_amount=latest.minimum_allowed_amount,
                is_protected=is_protected,
                is_essential=is_protected or latest.flexibility is Flexibility.FIXED,
            )
        )

    variable: list[VariableSpendingProfile] = []
    by_category: dict[Category, list[LedgerEntry]] = defaultdict(list)
    for entry in history:
        if (
            entry.direction is Direction.DEBIT
            and entry.category in VARIABLE_CATEGORIES
            and entry.event_id not in consumed
        ):
            by_category[entry.category].append(entry)
    for category, entries in sorted(by_category.items(), key=lambda item: item[0].value):
        if len(entries) < MIN_VARIABLE_OCCURRENCES:
            continue
        dates = [entry.cash_date for entry in entries]
        period, _ = _modal((later - earlier).days for earlier, later in pairwise(dates))
        if not 1 <= period <= MAX_VARIABLE_PERIOD_DAYS:
            continue
        with decimal_policy():
            total = sum((entry.amount for entry in entries), start=Decimal(0))
        variable.append(
            VariableSpendingProfile(
                category=category,
                window_start=dates[0],
                window_end=dates[-1],
                observed_total=total,
                observation_count=len(entries),
                projected_amount_per_period=mean_amount(entry.amount for entry in entries),
                period_days=period,
                last_seen=dates[-1],
                amount_basis=AmountBasis.MEAN,
                is_essential=category in protected,
            )
        )
    return DetectRecurrenceOutput(
        request_id=payload.request_id, recurring=tuple(recurring), variable_spending=tuple(variable)
    )
