from datetime import date, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from buy_or_wait.schemas.enums import (
    AgentRole,
    PaymentMethod,
    PlannerPhase,
    RoutingReason,
    TerminalOutcome,
    ToolCallStatus,
    ToolName,
    TransitionReason,
)
from buy_or_wait.schemas.state import (
    IllegalTransitionError,
    PlannerDirective,
    PlannerState,
    ReflectionBudget,
    ReflectionLimitExceededError,
    StepLimitExceededError,
    ToolCallRecord,
)
from buy_or_wait.schemas.tools import (
    CandidatePlan,
    DailyBalance,
    DecisionVerificationOutput,
    DetectRecurrenceOutput,
    EarliestFullPaymentOutput,
    EvaluatedPlan,
    ForecastBalancesOutput,
    NormalizeLedgerOutput,
    PlanRankingOutput,
    ResolveEvidenceOutput,
    SafeAmountOutput,
)
from tests.conftest import Dataset, sample_decision

REQUEST_ID = "request_01"
STAGE = TransitionReason.STAGE_COMPLETED
DIGEST = "0" * 64


def _advance_to_verified(state: PlannerState, dataset: Dataset) -> PlannerState:
    context = dataset.context_for(REQUEST_ID)
    request, profile = context.request, context.profile
    start = request.request_date
    balance = profile.current_available_balance
    decision = sample_decision(dataset, REQUEST_ID)
    candidate = CandidatePlan(
        candidate_id="full_payment",
        method=PaymentMethod.FULL_PAYMENT,
        plan=decision.payment_plan,
        payment_option_id="payment_option_01",
        total_paid=request.requested_amount,
    )
    day = DailyBalance(
        balance_date=start,
        opening_balance=balance,
        total_credits=Decimal(0),
        total_debits=Decimal(0),
        intraday_low=balance,
        closing_balance=balance,
    )
    state = state.with_artifacts(context=context).transition(PlannerPhase.CONTEXT_LOADED, STAGE)
    state = state.with_artifacts(
        evidence=ResolveEvidenceOutput(request_id=REQUEST_ID, resolved=(), rejected=())
    ).transition(PlannerPhase.EVIDENCE_RESOLVED, STAGE)
    state = state.with_artifacts(
        ledger=NormalizeLedgerOutput(request_id=REQUEST_ID, entries=(), exclusions=()),
        recurrence=DetectRecurrenceOutput(
            request_id=REQUEST_ID, recurring=(), variable_spending=()
        ),
    ).transition(PlannerPhase.LEDGER_NORMALIZED, STAGE)
    state = state.with_artifacts(
        baseline_forecast=ForecastBalancesOutput(
            request_id=REQUEST_ID,
            start_date=start,
            end_date=start,
            minimum_balance=profile.minimum_balance_to_keep,
            days=(day,),
            lowest_balance=balance,
            lowest_balance_date=start,
            breaches_minimum=False,
        ),
        safe_amount=SafeAmountOutput(
            request_id=REQUEST_ID,
            requested_amount=request.requested_amount,
            minimum_headroom=balance - profile.minimum_balance_to_keep,
            binding_date=start,
            amount_safe_to_pay=request.requested_amount,
        ),
        earliest_full_payment=EarliestFullPaymentOutput(
            request_id=REQUEST_ID,
            search_start=start,
            search_end=start + timedelta(days=90),
            earliest_date=start,
        ),
    ).transition(PlannerPhase.FORECAST_READY, STAGE)
    return _redo_from_forecast(state, candidate, decision_start=start, dataset=dataset)


def _redo_from_forecast(
    state: PlannerState, candidate: CandidatePlan, *, decision_start: date, dataset: Dataset
) -> PlannerState:
    evaluation = EvaluatedPlan(
        candidate=candidate,
        is_safe=True,
        completes_by_deadline=True,
        lowest_balance=Decimal("33225.10"),
        lowest_balance_date=decision_start,
        violations=(),
    )
    if state.phase is PlannerPhase.FORECAST_READY:
        state = state.with_artifacts(
            plan_evaluations=(evaluation,),
            ranking=PlanRankingOutput(
                request_id=REQUEST_ID,
                ranked_candidate_ids=("full_payment",),
                selected_candidate_id="full_payment",
            ),
        ).transition(PlannerPhase.PLANS_EVALUATED, STAGE)
    state = state.with_artifacts(draft_decision=sample_decision(dataset, REQUEST_ID)).transition(
        PlannerPhase.DECISION_DRAFTED, STAGE
    )
    return state.with_artifacts(
        verification=DecisionVerificationOutput(request_id=REQUEST_ID, issues=(), is_valid=True)
    ).transition(PlannerPhase.VERIFIED, STAGE)


def _candidate(dataset: Dataset) -> CandidatePlan:
    decision = sample_decision(dataset, REQUEST_ID)
    return CandidatePlan(
        candidate_id="full_payment",
        method=PaymentMethod.FULL_PAYMENT,
        plan=decision.payment_plan,
        payment_option_id="payment_option_01",
        total_paid=decision.payment_plan.total(),
    )


def test_happy_path_finalizes_the_verified_draft(dataset: Dataset) -> None:
    state = _advance_to_verified(PlannerState(request_id=REQUEST_ID), dataset).finalize()
    assert state.phase is PlannerPhase.FINALIZED
    assert state.outcome is TerminalOutcome.FINALIZED
    assert state.final_decision == sample_decision(dataset, REQUEST_ID)
    assert state.is_terminal
    with pytest.raises(IllegalTransitionError):
        state.transition(PlannerPhase.REFLECTING, TransitionReason.VERIFICATION_FAILED)


def test_state_json_round_trip_is_reproducible(dataset: Dataset) -> None:
    state = _advance_to_verified(PlannerState(request_id=REQUEST_ID), dataset).finalize()
    restored = PlannerState.model_validate_json(state.model_dump_json())
    assert restored == state
    assert restored.digest() == state.digest()


def test_skipping_stages_is_illegal() -> None:
    with pytest.raises(IllegalTransitionError):
        PlannerState(request_id=REQUEST_ID).transition(PlannerPhase.FORECAST_READY, STAGE)


def test_stage_cannot_complete_without_its_artifact() -> None:
    with pytest.raises(ValidationError, match="missing artifacts"):
        PlannerState(request_id=REQUEST_ID).transition(PlannerPhase.CONTEXT_LOADED, STAGE)


def test_reflection_loop_is_bounded(dataset: Dataset) -> None:
    state = _advance_to_verified(
        PlannerState(request_id=REQUEST_ID, reflection=ReflectionBudget(limit=1)), dataset
    )
    assert state.can_reflect
    state = state.transition(PlannerPhase.REFLECTING, TransitionReason.VERIFICATION_FAILED)
    assert state.reflection.used == 1

    state = state.transition(PlannerPhase.PLANS_EVALUATED, TransitionReason.REFLECTION_RETRY)
    assert state.artifacts.draft_decision is None
    assert state.artifacts.verification is None
    assert state.artifacts.ranking is not None

    state = _redo_from_forecast(
        state, _candidate(dataset), decision_start=date(2024, 3, 3), dataset=dataset
    )
    assert state.phase is PlannerPhase.VERIFIED
    assert not state.can_reflect
    with pytest.raises(ReflectionLimitExceededError):
        state.transition(PlannerPhase.REFLECTING, TransitionReason.VERIFICATION_FAILED)

    failed = state.fail_safe(TransitionReason.REFLECTION_LIMIT_REACHED)
    assert failed.phase is PlannerPhase.FAILED
    assert failed.outcome is TerminalOutcome.SAFE_FALLBACK


def test_reflection_limit_is_capped() -> None:
    with pytest.raises(ValidationError):
        ReflectionBudget(limit=4)
    with pytest.raises(ValidationError):
        ReflectionBudget(limit=1, used=2)


def test_forged_reflection_usage_is_rejected() -> None:
    with pytest.raises(ValidationError, match="reflection budget usage"):
        PlannerState(request_id=REQUEST_ID, reflection=ReflectionBudget(limit=2, used=1))


def test_step_budget_is_enforced_but_abort_is_always_possible(dataset: Dataset) -> None:
    state = PlannerState(request_id=REQUEST_ID, max_steps=1).with_artifacts(
        context=dataset.context_for(REQUEST_ID)
    )
    state = state.transition(PlannerPhase.CONTEXT_LOADED, STAGE)
    with pytest.raises(StepLimitExceededError):
        state.transition(PlannerPhase.EVIDENCE_RESOLVED, STAGE)
    assert state.fail_safe(TransitionReason.STEP_LIMIT_REACHED).is_terminal


def test_directives_follow_the_routing_table() -> None:
    state = PlannerState(request_id=REQUEST_ID)
    directive = PlannerDirective(
        step=0,
        phase=PlannerPhase.INITIALIZED,
        next_agent=AgentRole.CONTEXT_LOADER,
        tool=ToolName.LOAD_USER_CONTEXT,
        reason=RoutingReason.NEXT_STAGE,
    )
    assert state.record_directive(directive).directives == (directive,)
    with pytest.raises(ValidationError, match="not routable"):
        PlannerDirective(
            step=0,
            phase=PlannerPhase.INITIALIZED,
            next_agent=AgentRole.VERIFIER,
            reason=RoutingReason.NEXT_STAGE,
        )


def test_tool_calls_respect_agent_permissions() -> None:
    with pytest.raises(ValidationError, match="not permitted"):
        ToolCallRecord(
            call_id="call_1",
            step=0,
            agent=AgentRole.FORECASTER,
            tool=ToolName.VERIFY_DECISION,
            status=ToolCallStatus.SUCCEEDED,
            input_digest=DIGEST,
            output_digest=DIGEST,
        )
    record = ToolCallRecord(
        call_id="call_1",
        step=0,
        agent=AgentRole.CONTEXT_LOADER,
        tool=ToolName.LOAD_USER_CONTEXT,
        status=ToolCallStatus.FAILED,
        input_digest=DIGEST,
        error_code="dataset_row_invalid",
    )
    state = PlannerState(request_id=REQUEST_ID).record_tool_call(record)
    assert state.tool_calls == (record,)
    with pytest.raises(IllegalTransitionError, match="does not hold control"):
        state.record_tool_call(
            record.evolve(agent=AgentRole.LEDGER_ANALYST, tool=ToolName.NORMALIZE_LEDGER)
        )
