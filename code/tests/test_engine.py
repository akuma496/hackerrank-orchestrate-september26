from datetime import date, timedelta
from decimal import Decimal

import pytest

from buy_or_wait.engine.errors import MissingExchangeRateError
from buy_or_wait.engine.forecast import (
    compute_safe_amount,
    find_earliest_full_payment_date,
    forecast_context,
)
from buy_or_wait.engine.ledger import convert_currency, normalize_ledger, rate_index
from buy_or_wait.engine.plans import rank_plans
from buy_or_wait.engine.policy import evaluate_request
from buy_or_wait.schemas.entities import ExchangeRate, FinancialEvent
from buy_or_wait.schemas.enums import (
    AmountSource,
    Category,
    Currency,
    Direction,
    EventStatus,
    EvidenceKind,
    EvidenceSource,
    ExclusionReason,
    Flexibility,
    PaymentMethod,
    QuoteVerification,
)
from buy_or_wait.schemas.evidence import ResolvedEvidence
from buy_or_wait.schemas.tools import (
    CurrencyConversionInput,
    DetectRecurrenceOutput,
    EarliestFullPaymentInput,
    ForecastContext,
    LedgerEntry,
    NormalizeLedgerInput,
    NormalizeLedgerOutput,
    PlanRankingInput,
    ResolveEvidenceOutput,
    SafeAmountInput,
)
from tests.conftest import Dataset

RID = "request_900"


def _event(event_id: str, **overrides: str) -> FinancialEvent:
    row = {
        "event_id": event_id,
        "user_id": "user_900",
        "event_type": "expense",
        "description": "Test record",
        "category": "shopping",
        "direction": "debit",
        "amount": "100",
        "currency": "INR",
        "event_date": "2024-03-01",
        "settlement_date": "2024-03-01",
        "status": "settled",
        "linked_event_id": "",
        "flexibility": "fixed",
        "minimum_allowed_amount": "",
    }
    row.update(overrides)
    return FinancialEvent.model_validate(row)


def _no_evidence(request_id: str) -> ResolveEvidenceOutput:
    return ResolveEvidenceOutput(request_id=request_id, resolved=(), rejected=())


def test_missing_exchange_rate_is_an_error() -> None:
    with pytest.raises(MissingExchangeRateError):
        convert_currency(
            CurrencyConversionInput(
                request_id=RID,
                amount=Decimal("10"),
                source_currency=Currency.USD,
                target_currency=Currency.INR,
                settlement_date=date(2024, 3, 16),
            ),
            rates={},
        )


def test_conversion_uses_exact_date_direction_and_bankers_rounding() -> None:
    rates = rate_index(
        [
            ExchangeRate.model_validate(
                {
                    "rate_date": "2024-03-15",
                    "from_currency": "USD",
                    "to_currency": "EUR",
                    "rate": "0.5",
                }
            )
        ]
    )
    converted = convert_currency(
        CurrencyConversionInput(
            request_id=RID,
            amount=Decimal("10.05"),
            source_currency=Currency.USD,
            target_currency=Currency.EUR,
            settlement_date=date(2024, 3, 15),
        ),
        rates,
    )
    assert converted.converted_amount == Decimal("5.02")
    with pytest.raises(MissingExchangeRateError):
        convert_currency(
            CurrencyConversionInput(
                request_id=RID,
                amount=Decimal("10"),
                source_currency=Currency.EUR,
                target_currency=Currency.USD,
                settlement_date=date(2024, 3, 15),
            ),
            rates,
        )


def test_ledger_inclusion_and_exclusion_rules() -> None:
    events = (
        _event("event_1"),
        _event("event_2", status="failed"),
        _event("event_3", status="cancelled"),
        _event(
            "event_4",
            event_type="refund",
            direction="credit",
            status="pending",
            settlement_date="2024-03-05",
        ),
        _event(
            "event_5",
            event_type="investment_valuation",
            direction="non_cash",
            status="unrealized",
            settlement_date="",
            category="investment",
        ),
        _event(
            "event_6",
            status="pending",
            linked_event_id="event_1",
            event_date="2024-03-02",
            settlement_date="2024-03-04",
        ),
        _event("event_7", amount=""),
        _event("event_8", amount=""),
        _event("event_9", status="pending", settlement_date="2024-03-06", amount="55.5"),
    )
    evidence = ResolvedEvidence(
        claim_id="claim_image_08",
        source=EvidenceSource.IMAGE,
        source_id="image_08",
        kind=EvidenceKind.DOCUMENT_AMOUNT,
        related_event_id="event_8",
        amount=Decimal("704.05"),
        currency=Currency.INR,
        verification=QuoteVerification.IMAGE_TRANSCRIPTION,
    )
    output = normalize_ledger(
        NormalizeLedgerInput(
            request_id=RID,
            request_date=date(2024, 3, 3),
            home_currency=Currency.INR,
            events=tuple(reversed(events)),
            resolved_evidence=(evidence,),
            exchange_rates=(),
        )
    )
    reasons = {item.event_id: item.reason for item in output.exclusions}
    assert reasons == {
        "event_2": ExclusionReason.FAILED,
        "event_3": ExclusionReason.CANCELLED,
        "event_4": ExclusionReason.PENDING_CREDIT,
        "event_5": ExclusionReason.UNREALIZED_NON_CASH,
        "event_6": ExclusionReason.DUPLICATE_RECORD,
        "event_7": ExclusionReason.UNRESOLVED_AMOUNT,
    }
    included = {entry.event_id: entry for entry in output.entries}
    assert set(included) == {"event_1", "event_8", "event_9"}
    assert included["event_8"].amount_source is AmountSource.IMAGE_EVIDENCE


def test_foreign_currency_without_rate_fails_the_ledger() -> None:
    with pytest.raises(MissingExchangeRateError):
        normalize_ledger(
            NormalizeLedgerInput(
                request_id=RID,
                request_date=date(2024, 3, 3),
                home_currency=Currency.INR,
                events=(_event("event_1", currency="USD"),),
                resolved_evidence=(),
                exchange_rates=(),
            )
        )


def _entry(event_id: str, day: date, direction: Direction, amount: str) -> LedgerEntry:
    category = Category.SALARY if direction is Direction.CREDIT else Category.RENT
    return LedgerEntry.model_validate(
        {
            "event_id": event_id,
            "cash_date": day,
            "direction": direction,
            "amount": Decimal(amount),
            "status": EventStatus.SCHEDULED,
            "category": category,
            "description": "Scheduled",
            "flexibility": Flexibility.FIXED,
            "amount_source": AmountSource.DATASET,
        }
    )


def test_safe_amount_and_earliest_date_on_a_hand_checked_trajectory() -> None:
    start = date(2024, 3, 1)
    ledger = NormalizeLedgerOutput(
        request_id=RID,
        entries=(
            _entry("event_1", start + timedelta(days=2), Direction.DEBIT, "300"),
            _entry("event_2", start + timedelta(days=4), Direction.CREDIT, "400"),
        ),
        exclusions=(),
    )
    context = ForecastContext(
        start_date=start,
        end_date=start + timedelta(days=9),
        home_currency=Currency.INR,
        opening_balance=Decimal("1000"),
        minimum_balance=Decimal("500"),
        ledger=ledger,
        recurrence=DetectRecurrenceOutput(request_id=RID, recurring=(), variable_spending=()),
    )
    _, forecast = forecast_context(context, RID)
    assert forecast.lowest_balance == Decimal("700")
    safe = compute_safe_amount(
        SafeAmountInput(
            request_id=RID,
            request_date=start,
            requested_amount=Decimal("600"),
            baseline_forecast=forecast,
        )
    )
    assert safe.amount_safe_to_pay == Decimal("200")
    earliest = find_earliest_full_payment_date(
        EarliestFullPaymentInput(
            request_id=RID,
            request_date=start,
            requested_amount=Decimal("600"),
            baseline_forecast=forecast,
        )
    )
    assert earliest.earliest_date == start + timedelta(days=4)


def test_unsafe_plans_are_never_ranked(dataset: Dataset) -> None:
    run = evaluate_request(dataset.context_for("request_20"), _no_evidence("request_20"))
    assert run.plan_evaluations
    assert all(not evaluation.is_safe for evaluation in run.plan_evaluations)
    assert (
        rank_plans(
            PlanRankingInput(request_id="request_20", evaluations=run.plan_evaluations)
        ).selected_candidate_id
        is None
    )
    assert run.decision.recommended_payment_method is PaymentMethod.NOT_RECOMMENDED


def test_decisions_are_independent_of_input_order(dataset: Dataset) -> None:
    context = dataset.context_for("request_21")
    shuffled_events = list(context.events)
    shuffled_events = shuffled_events[1::2] + shuffled_events[::-2]
    shuffled = context.evolve(
        events=tuple(shuffled_events), payment_options=tuple(reversed(context.payment_options))
    )
    first = evaluate_request(context, _no_evidence("request_21"))
    second = evaluate_request(shuffled, _no_evidence("request_21"))
    assert first.digest() == second.digest()


def test_sample_calibration_does_not_regress(dataset: Dataset) -> None:
    status = earliest = 0
    for row in dataset.sample_rows:
        decision = evaluate_request(
            dataset.context_for(row["request_id"]), _no_evidence(row["request_id"])
        ).decision.to_csv_row()
        status += decision["affordability_status"] == row["affordability_status"]
        earliest += (
            decision["earliest_date_for_full_payment"] == row["earliest_date_for_full_payment"]
        )
    assert status >= 16
    assert earliest >= 16
