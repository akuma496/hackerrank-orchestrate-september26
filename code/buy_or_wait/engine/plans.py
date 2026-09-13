"""Payment schedules, plan safety evaluation, spending-change search, and plan ranking.

A plan is *viable* only when its simulated trajectory never drops below the user's minimum
balance anywhere in the forecast window. Non-viable plans are never ranked or recommended.
"""

from datetime import date, timedelta
from decimal import Decimal
from typing import Final

from buy_or_wait.engine.forecast import forecast_context
from buy_or_wait.schemas.audit import OptionSchedule
from buy_or_wait.schemas.decision import PaymentPlan, ScheduledPayment, SpendingChange
from buy_or_wait.schemas.entities import FinancialProfile, PaymentOption
from buy_or_wait.schemas.enums import Direction, Flexibility, PaymentMethod, SpendingAction
from buy_or_wait.schemas.primitives import decimal_policy, id_ordinal
from buy_or_wait.schemas.tools import (
    AdjustableExpense,
    CandidatePlan,
    DetectRecurrenceOutput,
    EvaluatedPlan,
    ForecastContext,
    PaymentScheduleInput,
    PaymentScheduleOutput,
    PlanEvaluationInput,
    PlanEvaluationOutput,
    PlanRankingInput,
    PlanRankingOutput,
    PlanRankKey,
    SafetyViolation,
    SpendingChangeSearchInput,
    SpendingChangeSearchOutput,
)

MAX_CHANGES: Final[int] = 3


# ---------------------------------------------------------------------------
# build_payment_schedule
# ---------------------------------------------------------------------------
def build_payment_schedule(payload: PaymentScheduleInput) -> PaymentScheduleOutput:
    option = payload.option
    step = timedelta(days=option.payment_frequency_days or 0)
    payments = tuple(
        ScheduledPayment(
            payment_date=option.first_payment_date + step * index, amount=option.payment_amount
        )
        for index in range(option.number_of_payments)
    )
    return PaymentScheduleOutput(
        request_id=payload.request_id,
        schedule=OptionSchedule(
            payment_option_id=option.payment_option_id, plan=PaymentPlan(payments=payments)
        ),
        payment_method=option.payment_method,
        number_of_payments=option.number_of_payments,
        total_payable_amount=option.total_payable_amount,
    )


def installments_permitted(option: PaymentOption, profile: FinancialProfile) -> bool:
    """User accepts installments and the option has no more payments than months allowed."""
    return (
        option.payment_method is PaymentMethod.INSTALLMENTS
        and profile.accepts(PaymentMethod.INSTALLMENTS)
        and profile.max_installment_months is not None
        and option.number_of_payments <= profile.max_installment_months
    )


# ---------------------------------------------------------------------------
# evaluate_plan
# ---------------------------------------------------------------------------
def evaluate_plan(payload: PlanEvaluationInput) -> PlanEvaluationOutput:
    forecast = payload.forecast_with_plan
    minimum = forecast.minimum_balance
    with decimal_policy():
        violations = tuple(
            SafetyViolation(
                violation_date=day.balance_date,
                projected_balance=day.intraday_low,
                minimum_balance=minimum,
                shortfall=minimum - day.intraday_low,
            )
            for day in forecast.days
            if day.intraday_low < minimum
        )
    candidate = payload.candidate
    return PlanEvaluationOutput(
        request_id=payload.request_id,
        evaluation=EvaluatedPlan(
            candidate=candidate,
            is_safe=not violations,
            completes_by_deadline=(
                candidate.plan.payments[-1].payment_date <= payload.desired_completion_date
            ),
            lowest_balance=forecast.lowest_balance,
            lowest_balance_date=forecast.lowest_balance_date,
            violations=violations,
        ),
    )


def simulate_candidate(
    context: ForecastContext,
    request_id: str,
    candidate: CandidatePlan,
    request_date: date,
    desired_completion_date: date,
    requested_amount: Decimal,
) -> EvaluatedPlan:
    _, forecast = forecast_context(
        context,
        request_id,
        spending_changes=candidate.spending_changes,
        plan_payments=candidate.plan,
        payment_option_id=candidate.payment_option_id,
    )
    return evaluate_plan(
        PlanEvaluationInput(
            request_id=request_id,
            candidate=candidate,
            request_date=request_date,
            desired_completion_date=desired_completion_date,
            requested_amount=requested_amount,
            forecast_with_plan=forecast,
        )
    ).evaluation


# ---------------------------------------------------------------------------
# search_spending_changes
# ---------------------------------------------------------------------------
def adjustable_expenses(
    recurrence: DetectRecurrenceOutput, profile: FinancialProfile
) -> tuple[AdjustableExpense, ...]:
    """Flexible, unprotected recurring debits the user permits stopping or reducing."""
    expenses: list[AdjustableExpense] = []
    for series in recurrence.recurring:
        if (
            series.direction is not Direction.DEBIT
            or series.flexibility is Flexibility.FIXED
            or series.category in profile.expense_categories_to_protect
        ):
            continue
        actions: set[SpendingAction] = set()
        if (
            series.flexibility.can_stop
            and series.category in profile.expense_categories_user_is_willing_to_stop
        ):
            actions.add(SpendingAction.STOP)
        if (
            series.flexibility.can_reduce
            and series.category in profile.expense_categories_user_is_willing_to_reduce
            and series.minimum_allowed_amount is not None
            and series.minimum_allowed_amount < series.projected_amount
        ):
            actions.add(SpendingAction.REDUCE_TO)
        if actions:
            expenses.append(
                AdjustableExpense(
                    event_id=series.latest_event_id,
                    series_key=series.series_key,
                    category=series.category,
                    flexibility=series.flexibility,
                    current_amount=series.projected_amount,
                    minimum_allowed_amount=series.minimum_allowed_amount,
                    permitted_actions=frozenset(actions),
                )
            )
    return tuple(sorted(expenses, key=lambda expense: id_ordinal(expense.event_id)))


def _change_for(expense: AdjustableExpense) -> SpendingChange:
    """Prefer reducing to the permitted floor; stop only when reduction is not permitted."""
    floor = expense.minimum_allowed_amount
    if SpendingAction.REDUCE_TO in expense.permitted_actions and floor is not None and floor > 0:
        return SpendingChange(
            action=SpendingAction.REDUCE_TO, event_id=expense.event_id, new_amount=floor
        )
    return SpendingChange(action=SpendingAction.STOP, event_id=expense.event_id)


def _shortfall(evaluation: EvaluatedPlan) -> Decimal:
    with decimal_policy():
        return sum((violation.shortfall for violation in evaluation.violations), start=Decimal(0))


def search_spending_changes(payload: SpendingChangeSearchInput) -> SpendingChangeSearchOutput:
    """Greedy, deterministic search in event-id order; a change is kept only if it helps."""
    base = payload.candidate
    current = simulate_candidate(
        payload.context,
        payload.request_id,
        base,
        payload.request_date,
        payload.desired_completion_date,
        payload.requested_amount,
    )
    chosen: list[SpendingChange] = []
    savings = Decimal(0)
    expenses = sorted(payload.adjustable_expenses, key=lambda expense: id_ordinal(expense.event_id))
    for expense in expenses:
        if current.is_safe or len(chosen) >= payload.max_changes:
            break
        change = _change_for(expense)
        trial = base.evolve(
            candidate_id=f"{base.candidate_id}:changes",
            spending_changes=(*chosen, change),
        )
        evaluation = simulate_candidate(
            payload.context,
            payload.request_id,
            trial,
            payload.request_date,
            payload.desired_completion_date,
            payload.requested_amount,
        )
        if _shortfall(evaluation) < _shortfall(current):
            chosen.append(change)
            current = evaluation
            with decimal_policy():
                remaining = (
                    Decimal(0) if change.action is SpendingAction.STOP else change.new_amount
                )
                if remaining is None:
                    raise ValueError("reduce_to changes always carry a new amount")
                savings += expense.current_amount - remaining
    found = current.is_safe and bool(chosen)
    return SpendingChangeSearchOutput(
        request_id=payload.request_id,
        found=found,
        changes=tuple(chosen) if found else (),
        projected_savings=savings if found else Decimal(0),
    )


# ---------------------------------------------------------------------------
# rank_plans
# ---------------------------------------------------------------------------
def rank_plans(payload: PlanRankingInput) -> PlanRankingOutput:
    """Rank viable plans only, by the specification's lexicographic policy order."""
    viable = sorted(
        (evaluation for evaluation in payload.evaluations if evaluation.is_safe),
        key=lambda evaluation: (
            PlanRankKey.for_evaluation(evaluation).as_tuple(),
            evaluation.candidate.candidate_id,
        ),
    )
    ranked = tuple(evaluation.candidate.candidate_id for evaluation in viable)
    return PlanRankingOutput(
        request_id=payload.request_id,
        ranked_candidate_ids=ranked,
        selected_candidate_id=ranked[0] if ranked else None,
    )
