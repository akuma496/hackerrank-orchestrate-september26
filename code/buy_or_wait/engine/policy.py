"""Benchmark decision policy: build candidates, keep only viable ones, rank, and decide.

Candidate order is fixed and every collection is sorted before use, so a request always yields
the same decision regardless of input row order.
"""

from datetime import timedelta
from decimal import Decimal

from buy_or_wait.engine.explain import money, render_explanation
from buy_or_wait.engine.forecast import (
    compute_safe_amount,
    find_earliest_full_payment_date,
    forecast_context,
)
from buy_or_wait.engine.ledger import detect_recurrence, normalize_ledger
from buy_or_wait.engine.plans import (
    adjustable_expenses,
    build_payment_schedule,
    installments_permitted,
    rank_plans,
    search_spending_changes,
    simulate_candidate,
)
from buy_or_wait.schemas.audit import OptionSchedule
from buy_or_wait.schemas.base import StrictModel
from buy_or_wait.schemas.decision import (
    DecisionRecord,
    PaymentPlan,
    ScheduledPayment,
    SpendingChange,
)
from buy_or_wait.schemas.enums import AffordabilityStatus, PaymentMethod, SpendingAction
from buy_or_wait.schemas.primitives import decimal_policy, id_ordinal
from buy_or_wait.schemas.tools import (
    DEFAULT_HORIZON_DAYS,
    CandidatePlan,
    DetectRecurrenceInput,
    DetectRecurrenceOutput,
    EarliestFullPaymentInput,
    EarliestFullPaymentOutput,
    EvaluatedPlan,
    ExplanationFacts,
    ExplanationInput,
    ForecastBalancesOutput,
    ForecastContext,
    NormalizeLedgerInput,
    NormalizeLedgerOutput,
    PaymentScheduleInput,
    PlanRankingInput,
    PlanRankingOutput,
    ResolveEvidenceOutput,
    SafeAmountInput,
    SafeAmountOutput,
    SpendingChangeSearchInput,
    UserFinancialContext,
)


class EngineRun(StrictModel):
    """Every deterministic artifact behind one decision (fills the planner's artifact slots)."""

    ledger: NormalizeLedgerOutput
    recurrence: DetectRecurrenceOutput
    baseline_forecast: ForecastBalancesOutput
    safe_amount: SafeAmountOutput
    earliest_full_payment: EarliestFullPaymentOutput
    option_schedules: tuple[OptionSchedule, ...]
    plan_evaluations: tuple[EvaluatedPlan, ...]
    ranking: PlanRankingOutput
    decision: DecisionRecord


def _select(
    request_id: str, evaluations: list[EvaluatedPlan], excluded: frozenset[str]
) -> tuple[PlanRankingOutput, EvaluatedPlan | None]:
    """Rank viable plans; select the best one that completes by the deadline and is not excluded.

    ``excluded`` holds candidates the verifier rejected on earlier proposals.
    """
    ranking = rank_plans(PlanRankingInput(request_id=request_id, evaluations=tuple(evaluations)))
    by_id = {evaluation.candidate.candidate_id: evaluation for evaluation in evaluations}
    for candidate_id in ranking.ranked_candidate_ids:
        evaluation = by_id[candidate_id]
        if candidate_id not in excluded and evaluation.completes_by_deadline:
            return ranking.evolve(selected_candidate_id=candidate_id), evaluation
    return ranking.evolve(selected_candidate_id=None), None


def evaluate_request(
    context: UserFinancialContext,
    evidence: ResolveEvidenceOutput,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    excluded_candidates: frozenset[str] = frozenset(),
) -> EngineRun:
    request, profile = context.request, context.profile
    request_id = request.request_id
    rq, rd, deadline = (
        request.requested_amount,
        request.request_date,
        request.desired_completion_date,
    )
    resolved = tuple(sorted(evidence.resolved, key=lambda item: item.claim_id))

    ledger = normalize_ledger(
        NormalizeLedgerInput(
            request_id=request_id,
            request_date=rd,
            home_currency=profile.home_currency,
            events=context.events,
            resolved_evidence=resolved,
            exchange_rates=context.exchange_rates,
        )
    )
    recurrence = detect_recurrence(
        DetectRecurrenceInput(
            request_id=request_id, request_date=rd, profile=profile, ledger=ledger
        )
    )
    fctx = ForecastContext(
        start_date=rd,
        end_date=rd + timedelta(days=horizon_days),
        home_currency=profile.home_currency,
        opening_balance=profile.current_available_balance,
        minimum_balance=profile.minimum_balance_to_keep,
        ledger=ledger,
        recurrence=recurrence,
        resolved_evidence=resolved,
    )
    _, baseline = forecast_context(fctx, request_id)
    safe = compute_safe_amount(
        SafeAmountInput(
            request_id=request_id, request_date=rd, requested_amount=rq, baseline_forecast=baseline
        )
    )
    earliest = find_earliest_full_payment_date(
        EarliestFullPaymentInput(
            request_id=request_id, request_date=rd, requested_amount=rq, baseline_forecast=baseline
        )
    )
    options = sorted(
        context.payment_options, key=lambda option: id_ordinal(option.payment_option_id)
    )
    schedules = {
        option.payment_option_id: build_payment_schedule(
            PaymentScheduleInput(request_id=request_id, option=option)
        )
        for option in options
    }

    candidates: list[CandidatePlan] = []
    full_option = next(o for o in options if o.payment_method is PaymentMethod.FULL_PAYMENT)
    full_today = CandidatePlan(
        candidate_id="full_payment",
        method=PaymentMethod.FULL_PAYMENT,
        plan=PaymentPlan(payments=(ScheduledPayment(payment_date=rd, amount=rq),)),
        payment_option_id=full_option.payment_option_id,
        total_paid=rq,
    )
    if profile.accepts(PaymentMethod.FULL_PAYMENT):
        candidates.append(full_today)
    safe_now, earliest_date = safe.amount_safe_to_pay, earliest.earliest_date
    if (
        profile.accepts(PaymentMethod.PARTIAL_PAYMENT)
        and request.allows_partial_payment
        and Decimal(0) < safe_now < rq
        and earliest_date is not None
        and rd < earliest_date <= deadline
    ):
        with decimal_policy():
            remainder = rq - safe_now
        candidates.append(
            CandidatePlan(
                candidate_id="partial_payment",
                method=PaymentMethod.PARTIAL_PAYMENT,
                plan=PaymentPlan(
                    payments=(
                        ScheduledPayment(payment_date=rd, amount=safe_now),
                        ScheduledPayment(payment_date=earliest_date, amount=remainder),
                    )
                ),
                total_paid=rq,
            )
        )
    installment_candidates: list[CandidatePlan] = []
    for option in options:
        if installments_permitted(option, profile):
            plan = schedules[option.payment_option_id].schedule.plan
            installment_candidates.append(
                CandidatePlan(
                    candidate_id=f"installments:{option.payment_option_id}",
                    method=PaymentMethod.INSTALLMENTS,
                    plan=plan,
                    payment_option_id=option.payment_option_id,
                    total_paid=plan.total(),
                )
            )
    candidates.extend(installment_candidates)
    if (
        profile.accepts(PaymentMethod.FULL_PAYMENT)
        and earliest_date is not None
        and rd < earliest_date <= deadline
    ):
        candidates.append(
            CandidatePlan(
                candidate_id="wait",
                method=PaymentMethod.WAIT,
                plan=PaymentPlan(
                    payments=(ScheduledPayment(payment_date=earliest_date, amount=rq),)
                ),
                total_paid=rq,
            )
        )

    evaluations = [
        simulate_candidate(fctx, request_id, candidate, rd, deadline, rq)
        for candidate in sorted(candidates, key=lambda c: c.candidate_id)
    ]
    ranking, selected = _select(request_id, evaluations, excluded_candidates)

    if selected is None:
        expenses = adjustable_expenses(recurrence, profile)
        change_bases = [
            c
            for c in (full_today, *installment_candidates)
            if c.method is PaymentMethod.INSTALLMENTS or profile.accepts(c.method)
        ]
        if expenses:
            for base in change_bases:
                search = search_spending_changes(
                    SpendingChangeSearchInput(
                        request_id=request_id,
                        candidate=base,
                        context=fctx,
                        request_date=rd,
                        desired_completion_date=deadline,
                        requested_amount=rq,
                        adjustable_expenses=expenses,
                    )
                )
                if search.found:
                    changed = base.evolve(
                        candidate_id=f"{base.candidate_id}:changes",
                        spending_changes=search.changes,
                    )
                    evaluations.append(
                        simulate_candidate(fctx, request_id, changed, rd, deadline, rq)
                    )
            ranking, selected = _select(request_id, evaluations, excluded_candidates)

    changes: tuple[SpendingChange, ...]
    if selected is None:
        method, status, plan, changes = (
            PaymentMethod.NOT_RECOMMENDED,
            AffordabilityStatus.NOT_AFFORDABLE,
            PaymentPlan(),
            (),
        )
    else:
        candidate = selected.candidate
        method, plan, changes = candidate.method, candidate.plan, candidate.spending_changes
        if method is PaymentMethod.WAIT:
            status = AffordabilityStatus.AFFORDABLE_LATER
        elif method is PaymentMethod.FULL_PAYMENT and not changes:
            status = AffordabilityStatus.AFFORDABLE_NOW
        else:
            status = AffordabilityStatus.AFFORDABLE_WITH_PLAN

    series_by_event = {series.latest_event_id: series for series in recurrence.recurring}
    labels: list[str] = []
    for change in changes:
        description = series_by_event[change.event_id].description.lower()
        if change.action is SpendingAction.STOP:
            labels.append(f"stop the {description}")
        elif change.new_amount is not None:
            labels.append(
                f"reduce the {description} to {money(profile.home_currency, change.new_amount)}"
            )
    explanation = render_explanation(
        ExplanationInput(
            request_id=request_id,
            facts=ExplanationFacts(
                currency=profile.home_currency,
                status=status,
                method=method,
                requested_amount=rq,
                amount_safe_to_pay=safe_now,
                minimum_balance=profile.minimum_balance_to_keep,
                plan=plan,
                earliest_date=earliest_date,
                desired_completion_date=deadline,
                lowest_projected_balance=baseline.lowest_balance,
                spending_change_labels=tuple(labels),
                partial_payment_available=(
                    request.allows_partial_payment
                    and profile.accepts(PaymentMethod.PARTIAL_PAYMENT)
                ),
                horizon_days=horizon_days,
            ),
        )
    )
    decision = DecisionRecord(
        request_id=request_id,
        amount_safe_to_pay=safe_now,
        affordability_status=status,
        recommended_payment_method=method,
        payment_plan=plan,
        earliest_date_for_full_payment=earliest_date,
        spending_changes_needed=changes,
        decision_explanation=explanation.text,
    )
    return EngineRun(
        ledger=ledger,
        recurrence=recurrence,
        baseline_forecast=baseline,
        safe_amount=safe,
        earliest_full_payment=earliest,
        option_schedules=tuple(
            schedules[o.payment_option_id].schedule
            for o in options
            if o.payment_method is PaymentMethod.INSTALLMENTS
        ),
        plan_evaluations=tuple(sorted(evaluations, key=lambda e: e.candidate.candidate_id)),
        ranking=ranking,
        decision=decision,
    )
