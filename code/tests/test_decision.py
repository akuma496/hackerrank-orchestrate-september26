from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from buy_or_wait.schemas.audit import AuditedDecision, audit_decision
from buy_or_wait.schemas.decision import (
    DecisionRecord,
    PaymentPlan,
    ScheduledPayment,
    SpendingChange,
    parse_spending_changes,
)
from buy_or_wait.schemas.enums import (
    AffordabilityStatus,
    IssueCode,
    PaymentMethod,
    SpendingAction,
)
from tests.conftest import Dataset, audit_context_for, sample_decision


def _plan(*entries: tuple[date, str]) -> PaymentPlan:
    return PaymentPlan(
        payments=tuple(ScheduledPayment(payment_date=d, amount=Decimal(a)) for d, a in entries)
    )


def test_plan_field_round_trip() -> None:
    text = "2025-08-08:15952906.67|2025-09-07:15952906.67|2025-10-07:15952906.67"
    plan = PaymentPlan.from_field(text)
    assert plan.to_field() == text
    assert plan.total() == Decimal("47858720.01")
    assert PaymentPlan.from_field("none").is_empty


@pytest.mark.parametrize(
    "text", ["2024-09-15:10|2024-09-04:5", "2024-09-04:5|2024-09-04:5", "2024-09-04:-5", "soon:5"]
)
def test_plan_rejects_malformed_or_unordered_entries(text: str) -> None:
    with pytest.raises((ValidationError, ValueError)):
        PaymentPlan.from_field(text)


def test_spending_change_contracts() -> None:
    changes = parse_spending_changes("stop:event_1815|reduce_to:event_1816:23.50")
    assert changes[1].new_amount == Decimal("23.50")
    with pytest.raises(ValueError, match="not both"):
        parse_spending_changes("stop:event_1|reduce_to:event_1:10")
    with pytest.raises(ValueError, match="at most 3"):
        parse_spending_changes("stop:event_1|stop:event_2|stop:event_3|stop:event_4")
    with pytest.raises(ValidationError):
        SpendingChange(action=SpendingAction.STOP, event_id="event_1", new_amount=Decimal(5))
    with pytest.raises(ValidationError):
        SpendingChange(action=SpendingAction.REDUCE_TO, event_id="event_1")


def _record(**overrides: object) -> DecisionRecord:
    base: dict[str, object] = {
        "request_id": "request_19",
        "amount_safe_to_pay": Decimal("28820"),
        "affordability_status": AffordabilityStatus.AFFORDABLE_WITH_PLAN,
        "recommended_payment_method": PaymentMethod.PARTIAL_PAYMENT,
        "payment_plan": _plan((date(2024, 9, 4), "28820"), (date(2024, 9, 15), "10840")),
        "earliest_date_for_full_payment": date(2024, 9, 15),
        "spending_changes_needed": (),
        "decision_explanation": "Pay part today and the rest later.",
    }
    base.update(overrides)
    return DecisionRecord.model_validate(base)


def test_valid_partial_payment_record() -> None:
    assert _record().payment_plan.payment_count == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"recommended_payment_method": PaymentMethod.WAIT},
        {"affordability_status": AffordabilityStatus.AFFORDABLE_NOW},
        {"amount_safe_to_pay": Decimal("1")},
        {"earliest_date_for_full_payment": date(2024, 9, 20)},
        {"earliest_date_for_full_payment": None},
        {
            "recommended_payment_method": PaymentMethod.NOT_RECOMMENDED,
            "affordability_status": AffordabilityStatus.NOT_AFFORDABLE,
        },
        {
            "recommended_payment_method": PaymentMethod.FULL_PAYMENT,
            "payment_plan": _plan((date(2024, 9, 4), "39660")),
        },
        {"decision_explanation": "line one\nline two"},
    ],
)
def test_incoherent_rows_are_unrepresentable(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _record(**overrides)


def test_audit_accepts_the_published_partial_payment(dataset: Dataset) -> None:
    context = audit_context_for(dataset, "request_19")
    assert audit_decision(_record(), context) == ()


def test_audit_rejects_installments_that_match_no_option(dataset: Dataset) -> None:
    original = sample_decision(dataset, "request_02")
    shifted = original.evolve(
        payment_plan=PaymentPlan.from_field(
            "2025-08-09:15952906.67|2025-09-08:15952906.67|2025-10-08:15952906.67"
        )
    )
    issues = audit_decision(shifted, audit_context_for(dataset, "request_02"))
    assert IssueCode.PLAN_NOT_IN_OPTIONS in {issue.code for issue in issues}
    with pytest.raises(ValidationError, match="plan_not_in_options"):
        AuditedDecision.certify(shifted, audit_context_for(dataset, "request_02"))


def test_audit_rejects_partial_when_request_forbids_it(dataset: Dataset) -> None:
    decision = _record(request_id="request_02")
    issues = audit_decision(decision, audit_context_for(dataset, "request_02"))
    codes = {issue.code for issue in issues}
    assert IssueCode.PARTIAL_NOT_ALLOWED in codes


def test_audit_rejects_changes_to_protected_or_unknown_events(dataset: Dataset) -> None:
    original = sample_decision(dataset, "request_06")
    rent_event = next(
        event
        for event in dataset.events
        if event.user_id == "user_06" and event.category.value == "rent"
    )
    protected = original.evolve(
        spending_changes_needed=(
            SpendingChange(action=SpendingAction.STOP, event_id=rent_event.event_id),
        )
    )
    unknown = original.evolve(
        spending_changes_needed=(SpendingChange(action=SpendingAction.STOP, event_id="event_1"),)
    )
    context = audit_context_for(dataset, "request_06")
    assert IssueCode.SPENDING_CHANGE_NOT_PERMITTED in {
        issue.code for issue in audit_decision(protected, context)
    }
    assert IssueCode.SPENDING_CHANGE_UNKNOWN_EVENT in {
        issue.code for issue in audit_decision(unknown, context)
    }


def test_audited_decision_rejects_forged_issue_lists(dataset: Dataset) -> None:
    decision = sample_decision(dataset, "request_01")
    context = audit_context_for(dataset, "request_01")
    forged = audit_decision(_record(request_id="request_01"), context)
    assert forged
    with pytest.raises(ValidationError, match="exactly the audit findings"):
        AuditedDecision(decision=decision, context=context, issues=forged)
