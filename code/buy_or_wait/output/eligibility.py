"""Deterministic eligibility matrix: which (status, method) outcomes are legal for a request.

* affordable_now + full_payment: the user accepts full payment; one payment of the full amount
  on request_date; amount_safe_to_pay equals the request; earliest date equals request_date.
* affordable_with_plan + partial_payment: allowed by both user and request; exactly two payments
  totalling the request; the earliest full-payment date exists and is on or before the deadline.
* affordable_with_plan + installments: the user accepts installments; the plan equals a supplied
  option within the month limit; the last payment is on or before the deadline.
* affordable_with_plan + full_payment with cuts: one to three permitted cuts on flexible,
  unprotected events; stop and reduce never target the same event.
* affordable_later + wait: the user accepts full payment; one full payment on the earliest date,
  after today, within 90 days, and on or before the deadline.
* not_affordable + not_recommended: no payments and no cuts (the fallback when nothing works).

Anything else is ineligible. Contextual output rules from :func:`audit_decision` also apply.
"""

from collections.abc import Callable, Mapping
from datetime import timedelta
from types import MappingProxyType
from typing import Final

from buy_or_wait.schemas.audit import (
    DecisionAuditContext,
    OptionSchedule,
    ValidationIssue,
    audit_decision,
)
from buy_or_wait.schemas.decision import MAX_SPENDING_CHANGES, DecisionRecord
from buy_or_wait.schemas.enums import (
    AffordabilityStatus,
    IssueCode,
    IssueSeverity,
    OutputColumn,
    PaymentMethod,
)
from buy_or_wait.schemas.tools import DEFAULT_HORIZON_DAYS, UserFinancialContext

Check = Callable[[DecisionRecord, UserFinancialContext, tuple[OptionSchedule, ...]], list[str]]


def _full_today(decision: DecisionRecord, context: UserFinancialContext) -> list[str]:
    request = context.request
    payments = decision.payment_plan.payments
    problems: list[str] = []
    if not context.profile.accepts(PaymentMethod.FULL_PAYMENT):
        problems.append("user does not accept full payment")
    if (
        len(payments) != 1
        or payments[0].payment_date != request.request_date
        or (payments[0].amount != request.requested_amount)
    ):
        problems.append("full payment must pay the requested amount on request_date")
    return problems


def _affordable_now(
    d: DecisionRecord, c: UserFinancialContext, _: tuple[OptionSchedule, ...]
) -> list[str]:
    problems = _full_today(d, c)
    if d.amount_safe_to_pay != c.request.requested_amount:
        problems.append("affordable_now requires the full amount to be safe today")
    if d.earliest_date_for_full_payment != c.request.request_date:
        problems.append("affordable_now requires earliest date = request_date")
    if d.spending_changes_needed:
        problems.append("affordable_now takes no spending changes")
    return problems


def _partial(
    d: DecisionRecord, c: UserFinancialContext, _: tuple[OptionSchedule, ...]
) -> list[str]:
    request, payments = c.request, d.payment_plan.payments
    problems: list[str] = []
    if not (c.profile.accepts(PaymentMethod.PARTIAL_PAYMENT) and request.allows_partial_payment):
        problems.append("partial payment must be allowed by both the user and the request")
    if len(payments) != 2 or d.payment_plan.total() != request.requested_amount:
        problems.append("partial payment needs exactly two payments totalling the request")
    earliest = d.earliest_date_for_full_payment
    if earliest is None or earliest > request.desired_completion_date:
        problems.append("earliest full-payment date must exist and be on or before the deadline")
    if not 0 < d.amount_safe_to_pay < request.requested_amount:
        problems.append("partial payment needs 0 < amount_safe_to_pay < requested_amount")
    return problems


def _installments(
    d: DecisionRecord, c: UserFinancialContext, schedules: tuple[OptionSchedule, ...]
) -> list[str]:
    profile, request = c.profile, c.request
    problems: list[str] = []
    if not profile.accepts(PaymentMethod.INSTALLMENTS) or profile.max_installment_months is None:
        return ["user does not accept installments"]
    options = {o.payment_option_id: o for o in c.payment_options}
    matches = [
        s
        for s in schedules
        if s.plan == d.payment_plan
        and s.payment_option_id in options
        and options[s.payment_option_id].payment_method is PaymentMethod.INSTALLMENTS
        and options[s.payment_option_id].number_of_payments <= profile.max_installment_months
    ]
    if not matches:
        problems.append("installments must match a supplied option within the month limit")
    payments = d.payment_plan.payments
    if not payments or payments[-1].payment_date > request.desired_completion_date:
        problems.append("installments must complete by the deadline")
    return problems


def _cuts_then_full(
    d: DecisionRecord, c: UserFinancialContext, _: tuple[OptionSchedule, ...]
) -> list[str]:
    problems = _full_today(d, c)
    changes = d.spending_changes_needed
    if not 1 <= len(changes) <= MAX_SPENDING_CHANGES:
        problems.append("a spending-change plan needs one to three changes")
    if len({change.event_id for change in changes}) != len(changes):
        problems.append("stop and reduce cannot target the same event")
    return problems


def _wait(d: DecisionRecord, c: UserFinancialContext, _: tuple[OptionSchedule, ...]) -> list[str]:
    request, earliest = c.request, d.earliest_date_for_full_payment
    problems: list[str] = []
    if not c.profile.accepts(PaymentMethod.FULL_PAYMENT):
        problems.append("wait requires the user to accept full payment")
    horizon_end = request.request_date + timedelta(days=DEFAULT_HORIZON_DAYS)
    if earliest is None or not (
        request.request_date < earliest <= min(horizon_end, request.desired_completion_date)
    ):
        problems.append("wait needs a full payment later, within 90 days, by the deadline")
    payments = d.payment_plan.payments
    if (
        len(payments) != 1
        or payments[0].payment_date != earliest
        or (payments[0].amount != request.requested_amount)
    ):
        problems.append("wait pays the requested amount once, on the earliest date")
    if d.spending_changes_needed:
        problems.append("wait takes no spending changes")
    return problems


def _not_recommended(
    d: DecisionRecord, c: UserFinancialContext, _: tuple[OptionSchedule, ...]
) -> list[str]:
    if not d.payment_plan.is_empty or d.spending_changes_needed:
        return ["not_recommended carries no payments and no spending changes"]
    return []


ELIGIBILITY_MATRIX: Final[Mapping[tuple[AffordabilityStatus, PaymentMethod, bool], Check]] = (
    MappingProxyType(
        {
            (
                AffordabilityStatus.AFFORDABLE_NOW,
                PaymentMethod.FULL_PAYMENT,
                False,
            ): _affordable_now,
            (
                AffordabilityStatus.AFFORDABLE_WITH_PLAN,
                PaymentMethod.PARTIAL_PAYMENT,
                False,
            ): _partial,
            (
                AffordabilityStatus.AFFORDABLE_WITH_PLAN,
                PaymentMethod.INSTALLMENTS,
                False,
            ): _installments,
            (
                AffordabilityStatus.AFFORDABLE_WITH_PLAN,
                PaymentMethod.INSTALLMENTS,
                True,
            ): _installments,
            (
                AffordabilityStatus.AFFORDABLE_WITH_PLAN,
                PaymentMethod.FULL_PAYMENT,
                True,
            ): _cuts_then_full,
            (AffordabilityStatus.AFFORDABLE_LATER, PaymentMethod.WAIT, False): _wait,
            (
                AffordabilityStatus.NOT_AFFORDABLE,
                PaymentMethod.NOT_RECOMMENDED,
                False,
            ): _not_recommended,
        }
    )
)


def check_eligibility(
    decision: DecisionRecord,
    context: UserFinancialContext,
    schedules: tuple[OptionSchedule, ...],
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> tuple[ValidationIssue, ...]:
    key = (
        decision.affordability_status,
        decision.recommended_payment_method,
        bool(decision.spending_changes_needed),
    )
    rule = ELIGIBILITY_MATRIX.get(key)
    problems = (
        ["outcome is not in the eligibility matrix"]
        if rule is None
        else rule(decision, context, schedules)
    )
    issues = [
        ValidationIssue(
            code=IssueCode.ELIGIBILITY_VIOLATION,
            severity=IssueSeverity.ERROR,
            column=OutputColumn.RECOMMENDED_PAYMENT_METHOD,
            detail=problem,
        )
        for problem in problems
    ]
    audit_context = DecisionAuditContext(
        request=context.request,
        profile=context.profile,
        events=context.events,
        payment_options=context.payment_options,
        option_schedules=schedules,
        horizon_end=context.request.request_date + timedelta(days=horizon_days),
    )
    issues.extend(audit_decision(decision, audit_context))
    return tuple(issues)
