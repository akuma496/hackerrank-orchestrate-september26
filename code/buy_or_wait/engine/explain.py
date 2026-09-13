"""Deterministic explanation templates grounded only in :class:`ExplanationFacts`."""

from datetime import date
from decimal import Decimal

from buy_or_wait.schemas.enums import (
    AffordabilityStatus,
    Currency,
    ExplanationTemplate,
    PaymentMethod,
)
from buy_or_wait.schemas.primitives import decimal_policy, quantize_money
from buy_or_wait.schemas.tools import ExplanationFacts, ExplanationInput, ExplanationOutput


def money(currency: Currency, amount: Decimal) -> str:
    value = quantize_money(amount)
    spec = ",.0f" if value == value.to_integral_value() else ",.2f"
    return f"{currency.value} {format(value, spec)}"


def long_date(value: date) -> str:
    return f"{value.day} {value:%B} {value.year}"


def _joined(labels: tuple[str, ...]) -> str:
    text = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + f" and {labels[-1]}"
    return text[:1].upper() + text[1:]


def render_explanation(payload: ExplanationInput) -> ExplanationOutput:
    facts: ExplanationFacts = payload.facts
    cur = facts.currency
    minimum = money(cur, facts.minimum_balance)
    requested = money(cur, facts.requested_amount)
    payments = facts.plan.payments
    prefix = (
        f"{_joined(facts.spending_change_labels)}, then " if facts.spending_change_labels else ""
    )

    if facts.method is PaymentMethod.FULL_PAYMENT and not facts.spending_change_labels:
        template = ExplanationTemplate.PAY_IN_FULL_TODAY
        text = (
            f"Pay {requested} today. This leaves at least {minimum} available over the next "
            f"{facts.horizon_days} days."
        )
    elif facts.method is PaymentMethod.FULL_PAYMENT:
        template = ExplanationTemplate.CHANGES_THEN_PAY_TODAY
        text = f"{prefix}pay {requested} today. This leaves at least {minimum} available."
    elif facts.method is PaymentMethod.PARTIAL_PAYMENT:
        template = ExplanationTemplate.PARTIAL_PAYMENT
        first, second = payments[0], payments[-1]
        text = (
            f"Pay {money(cur, first.amount)} today and the remaining {money(cur, second.amount)} "
            f"on {long_date(second.payment_date)}. This completes the full request and keeps the "
            f"{minimum} minimum protected."
        )
    elif facts.method is PaymentMethod.INSTALLMENTS:
        template = ExplanationTemplate.INSTALLMENTS
        lead = "use" if prefix else "Use"
        text = (
            f"{prefix}{lead} {len(payments)} installments of {money(cur, payments[0].amount)}, "
            f"starting {long_date(payments[0].payment_date)}. This leaves at least {minimum} "
            "available."
        )
    elif facts.method is PaymentMethod.WAIT and facts.earliest_date is not None:
        template = ExplanationTemplate.WAIT_THEN_PAY
        text = (
            f"Pay {requested} in full on {long_date(facts.earliest_date)}. Paying earlier would "
            f"take the balance below the {minimum} minimum."
        )
    elif facts.partial_payment_available and facts.amount_safe_to_pay > 0:
        template = ExplanationTemplate.NOT_AFFORDABLE_WITHIN_HORIZON
        text = (
            f"Do not proceed with the {requested} request. Although "
            f"{money(cur, facts.amount_safe_to_pay)} is available today, the full amount cannot "
            f"be completed safely within {facts.horizon_days} days."
        )
    else:
        template = ExplanationTemplate.NOT_AFFORDABLE_BY_DEADLINE
        text = (
            f"Do not make this payment by {long_date(facts.desired_completion_date)}. None of the "
            f"available options keeps the {minimum} minimum protected."
        )
    if (
        facts.status is AffordabilityStatus.NOT_AFFORDABLE
        and facts.method is not PaymentMethod.NOT_RECOMMENDED
    ):
        raise ValueError("not_affordable decisions must use not_recommended")
    with decimal_policy():
        return ExplanationOutput(request_id=payload.request_id, template=template, text=text)
