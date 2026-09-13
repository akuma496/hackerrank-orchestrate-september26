"""Conservative fallback decision: ``not_affordable`` / ``not_recommended`` with no payments.

Used when verification fails three times, or when data is unavailable, conflicting, malformed,
or unsafe. Only tool-produced numbers are reused; when none exist the safe amount is zero.
"""

from datetime import date
from decimal import Decimal
from enum import StrEnum

from buy_or_wait.engine.explain import long_date, money
from buy_or_wait.schemas.decision import DecisionRecord, PaymentPlan
from buy_or_wait.schemas.entities import FinancialProfile, PurchaseRequest
from buy_or_wait.schemas.enums import AffordabilityStatus, PaymentMethod


class FallbackReason(StrEnum):
    VERIFICATION_LIMIT = "verification_limit"
    PROCESSING_ERROR = "processing_error"


def fallback_decision(
    request: PurchaseRequest,
    profile: FinancialProfile | None,
    reason: FallbackReason,
    safe_amount: Decimal | None = None,
    earliest_date: date | None = None,
) -> DecisionRecord:
    safe = Decimal(0) if safe_amount is None else min(max(safe_amount, Decimal(0)), request.requested_amount)
    deadline = long_date(request.desired_completion_date)
    if reason is FallbackReason.VERIFICATION_LIMIT and profile is not None:
        text = (
            f"Do not make this payment by {deadline}. No plan passed verification, so the "
            f"{money(profile.home_currency, profile.minimum_balance_to_keep)} minimum cannot be "
            "guaranteed."
        )
    else:
        text = (
            f"Do not make this payment by {deadline}. The request could not be evaluated safely "
            "from the available records, so no payment is recommended."
        )
    return DecisionRecord(
        request_id=request.request_id,
        amount_safe_to_pay=safe if reason is FallbackReason.VERIFICATION_LIMIT else Decimal(0),
        affordability_status=AffordabilityStatus.NOT_AFFORDABLE,
        recommended_payment_method=PaymentMethod.NOT_RECOMMENDED,
        payment_plan=PaymentPlan(),
        earliest_date_for_full_payment=(
            earliest_date if reason is FallbackReason.VERIFICATION_LIMIT else None
        ),
        spending_changes_needed=(),
        decision_explanation=text,
    )
