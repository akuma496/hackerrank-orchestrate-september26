"""Conservative fallback decision: ``not_affordable`` / ``not_recommended`` with no payments.

Used when verification fails three times, or when data is unavailable, conflicting, malformed,
or unsafe. Only tool-produced numbers are reused; for processing errors the safe amount is zero.
The explanation comes from the deterministic fallback templates.
"""

from datetime import date
from decimal import Decimal
from enum import StrEnum

from buy_or_wait.engine.explain import long_date
from buy_or_wait.output.templates import FallbackTemplate, render_fallback_explanation
from buy_or_wait.schemas.decision import DecisionRecord, PaymentPlan
from buy_or_wait.schemas.enums import AffordabilityStatus, PaymentMethod
from buy_or_wait.schemas.tools import UserFinancialContext


class FallbackReason(StrEnum):
    VERIFICATION_LIMIT = "verification_limit"
    PROCESSING_ERROR = "processing_error"


def fallback_decision(
    context: UserFinancialContext,
    reason: FallbackReason,
    safe_amount: Decimal | None = None,
    earliest_date: date | None = None,
) -> DecisionRecord:
    request = context.request
    verified_limit = reason is FallbackReason.VERIFICATION_LIMIT
    safe = Decimal(0)
    if verified_limit and safe_amount is not None:
        safe = min(max(safe_amount, Decimal(0)), request.requested_amount)
    template = (
        FallbackTemplate.VERIFICATION_LIMIT if verified_limit else FallbackTemplate.PROCESSING_ERROR
    )
    return DecisionRecord(
        request_id=request.request_id,
        amount_safe_to_pay=safe,
        affordability_status=AffordabilityStatus.NOT_AFFORDABLE,
        recommended_payment_method=PaymentMethod.NOT_RECOMMENDED,
        payment_plan=PaymentPlan(),
        earliest_date_for_full_payment=earliest_date if verified_limit else None,
        spending_changes_needed=(),
        decision_explanation=render_fallback_explanation(
            template, context, long_date(request.desired_completion_date)
        ),
    )
