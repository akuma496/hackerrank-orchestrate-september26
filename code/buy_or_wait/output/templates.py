"""String-template synthesis: every explanation is a pure function of the decision and context.

The template is chosen by the (status, method, spending-change) outcome and filled only with
values already present in the decision row or the validated dataset records. The batch layer
re-renders each explanation and rejects any row whose text differs, so no free-form or LLM text
can reach ``output.csv``.
"""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from buy_or_wait.engine.explain import long_date, money, render_explanation
from buy_or_wait.schemas.decision import DecisionRecord
from buy_or_wait.schemas.enums import (
    AffordabilityStatus,
    ExplanationTemplate,
    PaymentMethod,
    SpendingAction,
)
from buy_or_wait.schemas.tools import (
    DEFAULT_HORIZON_DAYS,
    ExplanationFacts,
    ExplanationInput,
    UserFinancialContext,
)


class FallbackTemplate(StrEnum):
    VERIFICATION_LIMIT = "fallback_verification_limit"
    PROCESSING_ERROR = "fallback_processing_error"


TEMPLATE_BY_OUTCOME: Final[
    Mapping[tuple[AffordabilityStatus, PaymentMethod, bool], frozenset[ExplanationTemplate]]
] = MappingProxyType(
    {
        (AffordabilityStatus.AFFORDABLE_NOW, PaymentMethod.FULL_PAYMENT, False): frozenset(
            {ExplanationTemplate.PAY_IN_FULL_TODAY}
        ),
        (AffordabilityStatus.AFFORDABLE_WITH_PLAN, PaymentMethod.FULL_PAYMENT, True): frozenset(
            {ExplanationTemplate.CHANGES_THEN_PAY_TODAY}
        ),
        (AffordabilityStatus.AFFORDABLE_WITH_PLAN, PaymentMethod.PARTIAL_PAYMENT, False): frozenset(
            {ExplanationTemplate.PARTIAL_PAYMENT}
        ),
        (AffordabilityStatus.AFFORDABLE_WITH_PLAN, PaymentMethod.INSTALLMENTS, False): frozenset(
            {ExplanationTemplate.INSTALLMENTS}
        ),
        (AffordabilityStatus.AFFORDABLE_WITH_PLAN, PaymentMethod.INSTALLMENTS, True): frozenset(
            {ExplanationTemplate.INSTALLMENTS}
        ),
        (AffordabilityStatus.AFFORDABLE_LATER, PaymentMethod.WAIT, False): frozenset(
            {ExplanationTemplate.WAIT_THEN_PAY}
        ),
        (AffordabilityStatus.NOT_AFFORDABLE, PaymentMethod.NOT_RECOMMENDED, False): frozenset(
            {
                ExplanationTemplate.NOT_AFFORDABLE_BY_DEADLINE,
                ExplanationTemplate.NOT_AFFORDABLE_WITHIN_HORIZON,
            }
        ),
    }
)


def spending_change_labels(
    decision: DecisionRecord, context: UserFinancialContext
) -> tuple[str, ...]:
    """``stop the <description>`` / ``reduce the <description> to <CUR amount>`` phrases."""
    descriptions = {event.event_id: event.description.lower() for event in context.events}
    currency = context.profile.home_currency
    labels: list[str] = []
    for change in decision.spending_changes_needed:
        description = descriptions.get(change.event_id)
        if description is None:
            raise ValueError("spending change references an unknown event")
        if change.action is SpendingAction.STOP or change.new_amount is None:
            labels.append(f"stop the {description}")
        else:
            labels.append(f"reduce the {description} to {money(currency, change.new_amount)}")
    return tuple(labels)


def render_decision_explanation(
    decision: DecisionRecord,
    context: UserFinancialContext,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> tuple[ExplanationTemplate, str]:
    request, profile = context.request, context.profile
    output = render_explanation(
        ExplanationInput(
            request_id=request.request_id,
            facts=ExplanationFacts(
                currency=profile.home_currency,
                status=decision.affordability_status,
                method=decision.recommended_payment_method,
                requested_amount=request.requested_amount,
                amount_safe_to_pay=decision.amount_safe_to_pay,
                minimum_balance=profile.minimum_balance_to_keep,
                plan=decision.payment_plan,
                earliest_date=decision.earliest_date_for_full_payment,
                desired_completion_date=request.desired_completion_date,
                spending_change_labels=spending_change_labels(decision, context),
                partial_payment_available=(
                    request.allows_partial_payment
                    and profile.accepts(PaymentMethod.PARTIAL_PAYMENT)
                ),
                horizon_days=horizon_days,
            ),
        )
    )
    return output.template, output.text


def render_fallback_explanation(
    template: FallbackTemplate, context: UserFinancialContext | None, deadline_text: str | None
) -> str:
    lead = (
        f"Do not make this payment by {deadline_text}."
        if deadline_text
        else "Do not make this payment."
    )
    if template is FallbackTemplate.VERIFICATION_LIMIT and context is not None:
        minimum = money(context.profile.home_currency, context.profile.minimum_balance_to_keep)
        return f"{lead} No plan passed verification, so the {minimum} minimum cannot be guaranteed."
    return (
        f"{lead} The request could not be evaluated safely from the available records, so no "
        "payment is recommended."
    )


def expected_explanations(
    decision: DecisionRecord,
    context: UserFinancialContext,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> frozenset[str]:
    """Every text the synthesis layer could legitimately emit for this exact decision."""
    template, text = render_decision_explanation(decision, context, horizon_days)
    key = (
        decision.affordability_status,
        decision.recommended_payment_method,
        bool(decision.spending_changes_needed),
    )
    allowed = TEMPLATE_BY_OUTCOME.get(key, frozenset())
    texts = {text} if template in allowed else set()
    if decision.recommended_payment_method is PaymentMethod.NOT_RECOMMENDED:
        deadline = long_date(context.request.desired_completion_date)
        texts.update(
            render_fallback_explanation(item, context, deadline) for item in FallbackTemplate
        )
    return frozenset(texts)
