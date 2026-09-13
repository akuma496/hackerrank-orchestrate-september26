"""Contextual output gate: a decision is only writable once it passes :func:`audit_decision`.

``DecisionRecord`` enforces rules visible within one row. This module adds the rules that need
the request, profile, events, and supplied payment options. :class:`AuditedDecision` cannot be
constructed while any ``error`` issue remains, so the CSV writer can accept only audited rows.
"""

from datetime import date
from decimal import Decimal
from typing import Final, Self

from pydantic import model_validator

from buy_or_wait.schemas.base import StrictModel
from buy_or_wait.schemas.decision import DecisionRecord, PaymentPlan, ScheduledPayment
from buy_or_wait.schemas.entities import (
    FinancialEvent,
    FinancialProfile,
    PaymentOption,
    PurchaseRequest,
)
from buy_or_wait.schemas.enums import (
    AffordabilityStatus,
    Direction,
    IssueCode,
    IssueSeverity,
    OutputColumn,
    PaymentMethod,
    SpendingAction,
)
from buy_or_wait.schemas.primitives import IsoDate, PaymentOptionId, ShortText, decimal_policy

_ERROR: Final = IssueSeverity.ERROR
_WARNING: Final = IssueSeverity.WARNING


class ValidationIssue(StrictModel):
    """A PII-free finding: codes and column names only, never amounts."""

    code: IssueCode
    severity: IssueSeverity
    column: OutputColumn | None = None
    detail: ShortText


class OptionSchedule(StrictModel):
    """Concrete dated schedule of one supplied option (built by ``build_payment_schedule``)."""

    payment_option_id: PaymentOptionId
    plan: PaymentPlan

    @model_validator(mode="after")
    def _non_empty(self) -> Self:
        if self.plan.is_empty:
            raise ValueError("an option schedule has at least one payment")
        return self


class DecisionAuditContext(StrictModel):
    request: PurchaseRequest
    profile: FinancialProfile
    events: tuple[FinancialEvent, ...]
    payment_options: tuple[PaymentOption, ...]
    option_schedules: tuple[OptionSchedule, ...]
    horizon_end: IsoDate

    @model_validator(mode="after")
    def _context_is_consistent(self) -> Self:
        user_id = self.request.user_id
        if self.profile.user_id != user_id:
            raise ValueError("profile belongs to a different user")
        if any(event.user_id != user_id for event in self.events):
            raise ValueError("events must all belong to the requesting user")
        if any(option.request_id != self.request.request_id for option in self.payment_options):
            raise ValueError("payment options must all belong to the request")
        option_ids = {option.payment_option_id for option in self.payment_options}
        if any(schedule.payment_option_id not in option_ids for schedule in self.option_schedules):
            raise ValueError("schedules must reference supplied payment options")
        if self.horizon_end < self.request.request_date:
            raise ValueError("horizon_end cannot precede request_date")
        return self


def _issue(
    code: IssueCode, severity: IssueSeverity, column: OutputColumn | None, detail: str
) -> ValidationIssue:
    return ValidationIssue(code=code, severity=severity, column=column, detail=detail)


def _single_payment(payment_date: date, amount: Decimal) -> PaymentPlan:
    return PaymentPlan(payments=(ScheduledPayment(payment_date=payment_date, amount=amount),))


def audit_decision(
    decision: DecisionRecord, context: DecisionAuditContext
) -> tuple[ValidationIssue, ...]:
    """Check ``decision`` against every contextual output rule; returns all findings."""
    request, profile = context.request, context.profile
    method, status = decision.recommended_payment_method, decision.affordability_status
    plan, earliest = decision.payment_plan, decision.earliest_date_for_full_payment
    issues: list[ValidationIssue] = []

    if decision.request_id != request.request_id:
        issues.append(
            _issue(
                IssueCode.REQUEST_MISMATCH,
                _ERROR,
                OutputColumn.REQUEST_ID,
                "decision answers a different request",
            )
        )
    if decision.amount_safe_to_pay > request.requested_amount:
        issues.append(
            _issue(
                IssueCode.AMOUNT_OUT_OF_BOUNDS,
                _ERROR,
                OutputColumn.AMOUNT_SAFE_TO_PAY,
                "exceeds requested_amount",
            )
        )
    if earliest is not None and not request.request_date <= earliest <= context.horizon_end:
        issues.append(
            _issue(
                IssueCode.EARLIEST_DATE_OUT_OF_RANGE,
                _ERROR,
                OutputColumn.EARLIEST_DATE_FOR_FULL_PAYMENT,
                "outside [request_date, horizon_end]",
            )
        )
    if any(payment.payment_date < request.request_date for payment in plan.payments):
        issues.append(
            _issue(
                IssueCode.PLAN_BEFORE_REQUEST_DATE,
                _ERROR,
                OutputColumn.PAYMENT_PLAN,
                "payment scheduled before request_date",
            )
        )

    if status is AffordabilityStatus.AFFORDABLE_NOW and (
        earliest != request.request_date or decision.amount_safe_to_pay != request.requested_amount
    ):
        issues.append(
            _issue(
                IssueCode.STATUS_INCONSISTENT,
                _ERROR,
                OutputColumn.AFFORDABILITY_STATUS,
                "affordable_now requires full safety on request_date",
            )
        )

    required_method = PaymentMethod.FULL_PAYMENT if method is PaymentMethod.WAIT else method
    if method is not PaymentMethod.NOT_RECOMMENDED and not profile.accepts(required_method):
        issues.append(
            _issue(
                IssueCode.METHOD_NOT_ACCEPTED,
                _ERROR,
                OutputColumn.RECOMMENDED_PAYMENT_METHOD,
                "user does not consider this payment method",
            )
        )

    if method is PaymentMethod.FULL_PAYMENT and plan != _single_payment(
        request.request_date, request.requested_amount
    ):
        issues.append(
            _issue(
                IssueCode.PLAN_SHAPE_INVALID,
                _ERROR,
                OutputColumn.PAYMENT_PLAN,
                "full_payment pays requested_amount on request_date",
            )
        )

    if method is PaymentMethod.WAIT and earliest is not None:
        if plan != _single_payment(earliest, request.requested_amount):
            issues.append(
                _issue(
                    IssueCode.PLAN_SHAPE_INVALID,
                    _ERROR,
                    OutputColumn.PAYMENT_PLAN,
                    "wait pays requested_amount on the earliest date",
                )
            )
        if earliest <= request.request_date:
            issues.append(
                _issue(
                    IssueCode.STATUS_INCONSISTENT,
                    _ERROR,
                    OutputColumn.EARLIEST_DATE_FOR_FULL_PAYMENT,
                    "wait needs an earliest date after request_date",
                )
            )
        if earliest > request.desired_completion_date:
            issues.append(
                _issue(
                    IssueCode.DEADLINE_MISSED,
                    _WARNING,
                    OutputColumn.EARLIEST_DATE_FOR_FULL_PAYMENT,
                    "full payment only becomes safe after the deadline",
                )
            )

    if method is PaymentMethod.PARTIAL_PAYMENT:
        if not request.allows_partial_payment:
            issues.append(
                _issue(
                    IssueCode.PARTIAL_NOT_ALLOWED,
                    _ERROR,
                    OutputColumn.RECOMMENDED_PAYMENT_METHOD,
                    "request does not allow partial payment",
                )
            )
        if decision.amount_safe_to_pay >= request.requested_amount:
            issues.append(
                _issue(
                    IssueCode.AMOUNT_OUT_OF_BOUNDS,
                    _ERROR,
                    OutputColumn.AMOUNT_SAFE_TO_PAY,
                    "partial payment needs amount_safe_to_pay below the request",
                )
            )
        if plan.payments and plan.payments[0].payment_date != request.request_date:
            issues.append(
                _issue(
                    IssueCode.PLAN_SHAPE_INVALID,
                    _ERROR,
                    OutputColumn.PAYMENT_PLAN,
                    "first partial payment is due on request_date",
                )
            )
        if plan.total() != request.requested_amount:
            issues.append(
                _issue(
                    IssueCode.PLAN_TOTAL_MISMATCH,
                    _ERROR,
                    OutputColumn.PAYMENT_PLAN,
                    "partial payments must add up to requested_amount",
                )
            )
        if earliest is not None and earliest > request.desired_completion_date:
            issues.append(
                _issue(
                    IssueCode.DEADLINE_MISSED,
                    _ERROR,
                    OutputColumn.EARLIEST_DATE_FOR_FULL_PAYMENT,
                    "remaining payment falls after desired_completion_date",
                )
            )

    if method is PaymentMethod.INSTALLMENTS:
        installment_ids = {
            option.payment_option_id
            for option in context.payment_options
            if option.payment_method is PaymentMethod.INSTALLMENTS
        }
        if not any(
            schedule.plan == plan and schedule.payment_option_id in installment_ids
            for schedule in context.option_schedules
        ):
            issues.append(
                _issue(
                    IssueCode.PLAN_NOT_IN_OPTIONS,
                    _ERROR,
                    OutputColumn.PAYMENT_PLAN,
                    "installment plan must exactly match a supplied option",
                )
            )

    issues.extend(_audit_spending_changes(decision, context))
    return tuple(issues)


def _audit_spending_changes(
    decision: DecisionRecord, context: DecisionAuditContext
) -> list[ValidationIssue]:
    column = OutputColumn.SPENDING_CHANGES_NEEDED
    profile = context.profile
    events = {event.event_id: event for event in context.events}
    issues: list[ValidationIssue] = []
    for change in decision.spending_changes_needed:
        event = events.get(change.event_id)
        if event is None:
            issues.append(
                _issue(
                    IssueCode.SPENDING_CHANGE_UNKNOWN_EVENT,
                    _ERROR,
                    column,
                    "referenced event does not belong to the user",
                )
            )
            continue
        if change.action is SpendingAction.STOP:
            permitted = (
                event.flexibility.can_stop
                and event.category in profile.expense_categories_user_is_willing_to_stop
            )
        else:
            permitted = (
                event.flexibility.can_reduce
                and event.category in profile.expense_categories_user_is_willing_to_reduce
            )
        if (
            not permitted
            or event.direction is not Direction.DEBIT
            or event.category in profile.expense_categories_to_protect
        ):
            issues.append(
                _issue(
                    IssueCode.SPENDING_CHANGE_NOT_PERMITTED,
                    _ERROR,
                    column,
                    "event is protected, fixed, or outside permitted categories",
                )
            )
            continue
        if change.action is SpendingAction.REDUCE_TO and change.new_amount is not None:
            floor = event.minimum_allowed_amount
            if floor is not None and change.new_amount < floor:
                issues.append(
                    _issue(
                        IssueCode.SPENDING_CHANGE_BELOW_MINIMUM,
                        _ERROR,
                        column,
                        "reduced amount is below minimum_allowed_amount",
                    )
                )
            if event.amount is None or change.new_amount >= event.amount:
                issues.append(
                    _issue(
                        IssueCode.SPENDING_CHANGE_NOT_A_REDUCTION,
                        _ERROR,
                        column,
                        "reduced amount must be below the current amount",
                    )
                )
    return issues


class AuditedDecision(StrictModel):
    """A decision proven to satisfy every error-level output rule for its context."""

    decision: DecisionRecord
    context: DecisionAuditContext
    issues: tuple[ValidationIssue, ...]

    @model_validator(mode="after")
    def _passes_audit(self) -> Self:
        with decimal_policy():
            recomputed = audit_decision(self.decision, self.context)
        if recomputed != self.issues:
            raise ValueError("issues must be exactly the audit findings for this decision")
        blocking = sorted({issue.code.value for issue in recomputed if issue.severity is _ERROR})
        if blocking:
            raise ValueError(f"decision fails the output gate: {blocking}")
        return self

    @classmethod
    def certify(cls, decision: DecisionRecord, context: DecisionAuditContext) -> Self:
        return cls(decision=decision, context=context, issues=audit_decision(decision, context))
