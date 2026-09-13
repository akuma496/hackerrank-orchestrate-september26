"""Multi-agent state container for the planner, with a hard-bounded reflection loop.

The planner *routes*; it never computes. Every state change goes through :meth:`PlannerState.
transition` (or another ``with_*`` method), which returns a new, fully re-validated instance.
Illegal routes, tool calls outside an agent's permissions, exceeding the reflection budget, or
exceeding the step budget raise before any state is produced.

Pipeline::

    INITIALIZED -> CONTEXT_LOADED -> EVIDENCE_RESOLVED -> LEDGER_NORMALIZED -> FORECAST_READY
      -> PLANS_EVALUATED -> DECISION_DRAFTED -> VERIFIED -> FINALIZED
                                                    |
                                                    +-> REFLECTING -> (re-enter a stage) ...
    any non-terminal phase -> FAILED (deterministic safe fallback)
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Self

from pydantic import field_validator, model_validator

from buy_or_wait.schemas.base import StrictModel
from buy_or_wait.schemas.decision import DecisionRecord
from buy_or_wait.schemas.enums import (
    AgentRole,
    PlannerPhase,
    RoutingReason,
    TerminalOutcome,
    ToolCallStatus,
    ToolName,
    TransitionReason,
)
from buy_or_wait.schemas.primitives import MachineKey, NonNegativeCount, RequestId, Sha256Hex
from buy_or_wait.schemas.tools import (
    DecisionVerificationOutput,
    DetectRecurrenceOutput,
    EarliestFullPaymentOutput,
    EvaluatedPlan,
    ForecastBalancesOutput,
    NormalizeLedgerOutput,
    PlanRankingOutput,
    ResolveEvidenceOutput,
    SafeAmountOutput,
    UserFinancialContext,
)

MAX_REFLECTION_LIMIT: Final[int] = 3
DEFAULT_REFLECTION_LIMIT: Final[int] = 2
MAX_STEP_LIMIT: Final[int] = 128
DEFAULT_STEP_LIMIT: Final[int] = 40

_P = PlannerPhase

TERMINAL_PHASES: Final[frozenset[PlannerPhase]] = frozenset({_P.FINALIZED, _P.FAILED})

PHASE_ORDER: Final[tuple[PlannerPhase, ...]] = (
    _P.INITIALIZED,
    _P.CONTEXT_LOADED,
    _P.EVIDENCE_RESOLVED,
    _P.LEDGER_NORMALIZED,
    _P.FORECAST_READY,
    _P.PLANS_EVALUATED,
    _P.DECISION_DRAFTED,
    _P.VERIFIED,
)

REENTRY_PHASES: Final[frozenset[PlannerPhase]] = frozenset(
    {_P.EVIDENCE_RESOLVED, _P.LEDGER_NORMALIZED, _P.FORECAST_READY, _P.PLANS_EVALUATED}
)
"""Stages a reflection may resume from (context loading is never redone)."""

ALLOWED_TRANSITIONS: Final[Mapping[PlannerPhase, frozenset[PlannerPhase]]] = MappingProxyType(
    {
        _P.INITIALIZED: frozenset({_P.CONTEXT_LOADED, _P.FAILED}),
        _P.CONTEXT_LOADED: frozenset({_P.EVIDENCE_RESOLVED, _P.FAILED}),
        _P.EVIDENCE_RESOLVED: frozenset({_P.LEDGER_NORMALIZED, _P.FAILED}),
        _P.LEDGER_NORMALIZED: frozenset({_P.FORECAST_READY, _P.FAILED}),
        _P.FORECAST_READY: frozenset({_P.PLANS_EVALUATED, _P.FAILED}),
        _P.PLANS_EVALUATED: frozenset({_P.DECISION_DRAFTED, _P.FAILED}),
        _P.DECISION_DRAFTED: frozenset({_P.VERIFIED, _P.FAILED}),
        _P.VERIFIED: frozenset({_P.FINALIZED, _P.REFLECTING, _P.FAILED}),
        _P.REFLECTING: REENTRY_PHASES | {_P.FAILED},
        _P.FINALIZED: frozenset(),
        _P.FAILED: frozenset(),
    }
)

TOOL_PERMISSIONS: Final[Mapping[AgentRole, frozenset[ToolName]]] = MappingProxyType(
    {
        AgentRole.PLANNER: frozenset(),
        AgentRole.CONTEXT_LOADER: frozenset({ToolName.LOAD_USER_CONTEXT}),
        AgentRole.EVIDENCE_INTERPRETER: frozenset({ToolName.RESOLVE_EVIDENCE}),
        AgentRole.LEDGER_ANALYST: frozenset(
            {ToolName.CONVERT_CURRENCY, ToolName.NORMALIZE_LEDGER, ToolName.DETECT_RECURRENCE}
        ),
        AgentRole.FORECASTER: frozenset(
            {
                ToolName.PROJECT_CASH_FLOWS,
                ToolName.FORECAST_BALANCES,
                ToolName.COMPUTE_SAFE_AMOUNT,
                ToolName.FIND_EARLIEST_FULL_PAYMENT_DATE,
            }
        ),
        AgentRole.PLAN_STRATEGIST: frozenset(
            {
                ToolName.BUILD_PAYMENT_SCHEDULE,
                ToolName.PROJECT_CASH_FLOWS,
                ToolName.FORECAST_BALANCES,
                ToolName.EVALUATE_PLAN,
                ToolName.SEARCH_SPENDING_CHANGES,
                ToolName.RANK_PLANS,
            }
        ),
        AgentRole.EXPLAINER: frozenset({ToolName.RENDER_EXPLANATION}),
        AgentRole.VERIFIER: frozenset({ToolName.BUILD_PAYMENT_SCHEDULE, ToolName.VERIFY_DECISION}),
    }
)

ROUTES: Final[Mapping[PlannerPhase, frozenset[AgentRole]]] = MappingProxyType(
    {
        _P.INITIALIZED: frozenset({AgentRole.CONTEXT_LOADER}),
        _P.CONTEXT_LOADED: frozenset({AgentRole.EVIDENCE_INTERPRETER}),
        _P.EVIDENCE_RESOLVED: frozenset({AgentRole.LEDGER_ANALYST}),
        _P.LEDGER_NORMALIZED: frozenset({AgentRole.FORECASTER}),
        _P.FORECAST_READY: frozenset({AgentRole.PLAN_STRATEGIST}),
        _P.PLANS_EVALUATED: frozenset({AgentRole.EXPLAINER}),
        _P.DECISION_DRAFTED: frozenset({AgentRole.VERIFIER}),
        _P.VERIFIED: frozenset({AgentRole.PLANNER}),
        _P.REFLECTING: frozenset(
            {
                AgentRole.PLANNER,
                AgentRole.EVIDENCE_INTERPRETER,
                AgentRole.LEDGER_ANALYST,
                AgentRole.FORECASTER,
                AgentRole.PLAN_STRATEGIST,
            }
        ),
        _P.FINALIZED: frozenset(),
        _P.FAILED: frozenset(),
    }
)
"""Which agent the planner may hand control to from each phase."""

ARTIFACT_INTRODUCED_AT: Final[Mapping[str, PlannerPhase]] = MappingProxyType(
    {
        "context": _P.CONTEXT_LOADED,
        "evidence": _P.EVIDENCE_RESOLVED,
        "ledger": _P.LEDGER_NORMALIZED,
        "recurrence": _P.LEDGER_NORMALIZED,
        "baseline_forecast": _P.FORECAST_READY,
        "safe_amount": _P.FORECAST_READY,
        "earliest_full_payment": _P.FORECAST_READY,
        "ranking": _P.PLANS_EVALUATED,
        "draft_decision": _P.DECISION_DRAFTED,
        "verification": _P.VERIFIED,
    }
)
"""The phase that first requires each artifact; later phases require it cumulatively."""


class IllegalTransitionError(RuntimeError):
    """The requested phase change is not allowed by the planner state machine."""


class ReflectionLimitExceededError(RuntimeError):
    """The reflection budget is spent; the planner must fail over to the safe fallback."""


class StepLimitExceededError(RuntimeError):
    """The step budget is spent; the planner must fail over to the safe fallback."""


def _phase_rank(phase: PlannerPhase) -> int:
    return PHASE_ORDER.index(phase) if phase in PHASE_ORDER else len(PHASE_ORDER)


def _required_artifacts(phase: PlannerPhase) -> frozenset[str]:
    if phase is _P.FAILED:
        return frozenset()
    effective = _P.VERIFIED if phase in {_P.REFLECTING, _P.FINALIZED} else phase
    rank = _phase_rank(effective)
    return frozenset(
        name
        for name, introduced in ARTIFACT_INTRODUCED_AT.items()
        if _phase_rank(introduced) <= rank
    )


# ===========================================================================
# Records
# ===========================================================================
class ReflectionBudget(StrictModel):
    """Hard cap on verify -> reflect -> retry cycles."""

    limit: NonNegativeCount = DEFAULT_REFLECTION_LIMIT
    used: NonNegativeCount = 0

    @field_validator("limit")
    @classmethod
    def _limit_is_bounded(cls, value: int) -> int:
        if value > MAX_REFLECTION_LIMIT:
            raise ValueError(f"reflection limit cannot exceed {MAX_REFLECTION_LIMIT}")
        return value

    @model_validator(mode="after")
    def _within_budget(self) -> Self:
        if self.used > self.limit:
            raise ValueError("reflections used cannot exceed the limit")
        return self

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    def consume(self) -> Self:
        if self.exhausted:
            raise ReflectionLimitExceededError("reflection budget exhausted")
        return self.evolve(used=self.used + 1)


class PhaseTransition(StrictModel):
    step: NonNegativeCount
    from_phase: PlannerPhase
    to_phase: PlannerPhase
    reason: TransitionReason

    @model_validator(mode="after")
    def _allowed(self) -> Self:
        if self.to_phase not in ALLOWED_TRANSITIONS[self.from_phase]:
            raise ValueError("transition is not allowed by the state machine")
        return self


class ToolCallRecord(StrictModel):
    """Audit trail entry. Digests (not payloads) keep it reproducible and PII-free."""

    call_id: MachineKey
    step: NonNegativeCount
    agent: AgentRole
    tool: ToolName
    status: ToolCallStatus
    input_digest: Sha256Hex
    output_digest: Sha256Hex | None = None
    error_code: MachineKey | None = None

    @model_validator(mode="after")
    def _permitted_and_complete(self) -> Self:
        if self.tool not in TOOL_PERMISSIONS[self.agent]:
            raise ValueError("agent is not permitted to call this tool")
        succeeded = self.status is ToolCallStatus.SUCCEEDED
        if succeeded != (self.output_digest is not None):
            raise ValueError("exactly the successful calls carry an output digest")
        if succeeded == (self.error_code is not None):
            raise ValueError("exactly the unsuccessful calls carry an error code")
        return self


class PlannerDirective(StrictModel):
    """One routing decision. Codes only: the planner emits no numbers and no free text."""

    step: NonNegativeCount
    phase: PlannerPhase
    next_agent: AgentRole
    tool: ToolName | None = None
    reason: RoutingReason

    @model_validator(mode="after")
    def _route_is_allowed(self) -> Self:
        if self.next_agent not in ROUTES[self.phase]:
            raise ValueError("next_agent is not routable from this phase")
        if self.tool is not None and self.tool not in TOOL_PERMISSIONS[self.next_agent]:
            raise ValueError("directive names a tool the agent may not call")
        return self


class PlannerArtifacts(StrictModel):
    """Typed slots for tool outputs; only deterministic tools fill them."""

    context: UserFinancialContext | None = None
    evidence: ResolveEvidenceOutput | None = None
    ledger: NormalizeLedgerOutput | None = None
    recurrence: DetectRecurrenceOutput | None = None
    baseline_forecast: ForecastBalancesOutput | None = None
    safe_amount: SafeAmountOutput | None = None
    earliest_full_payment: EarliestFullPaymentOutput | None = None
    plan_evaluations: tuple[EvaluatedPlan, ...] = ()
    ranking: PlanRankingOutput | None = None
    draft_decision: DecisionRecord | None = None
    verification: DecisionVerificationOutput | None = None

    @model_validator(mode="after")
    def _cross_artifact_integrity(self) -> Self:
        if self.ranking is not None:
            evaluated_ids = {item.candidate.candidate_id for item in self.plan_evaluations}
            if set(self.ranking.ranked_candidate_ids) - evaluated_ids:
                raise ValueError("ranking references candidates that were not evaluated")
        request_ids = {
            artifact.request_id
            for artifact in (
                self.context,
                self.evidence,
                self.ledger,
                self.recurrence,
                self.baseline_forecast,
                self.safe_amount,
                self.earliest_full_payment,
                self.ranking,
                self.draft_decision,
                self.verification,
            )
            if artifact is not None
        }
        if len(request_ids) > 1:
            raise ValueError("all artifacts must describe the same request")
        return self

    def missing(self, names: frozenset[str]) -> frozenset[str]:
        return frozenset(name for name in names if getattr(self, name) is None)

    def cleared_after(self, phase: PlannerPhase) -> Self:
        """Drop artifacts introduced after ``phase`` (used when a reflection re-enters it)."""
        rank = _phase_rank(phase)
        stale: dict[str, object] = {
            name: None
            for name, introduced in ARTIFACT_INTRODUCED_AT.items()
            if _phase_rank(introduced) > rank
        }
        if _phase_rank(_P.PLANS_EVALUATED) > rank:
            stale["plan_evaluations"] = ()
        return self.evolve(**stale)


# ===========================================================================
# State container
# ===========================================================================
class PlannerState(StrictModel):
    """Immutable, self-validating state of one request's multi-agent run."""

    request_id: RequestId
    phase: PlannerPhase = _P.INITIALIZED
    step: NonNegativeCount = 0
    max_steps: NonNegativeCount = DEFAULT_STEP_LIMIT
    reflection: ReflectionBudget = ReflectionBudget()
    transitions: tuple[PhaseTransition, ...] = ()
    directives: tuple[PlannerDirective, ...] = ()
    tool_calls: tuple[ToolCallRecord, ...] = ()
    artifacts: PlannerArtifacts = PlannerArtifacts()
    outcome: TerminalOutcome | None = None
    final_decision: DecisionRecord | None = None

    @field_validator("max_steps")
    @classmethod
    def _step_limit_is_bounded(cls, value: int) -> int:
        if not 1 <= value <= MAX_STEP_LIMIT:
            raise ValueError(f"max_steps must be within 1..{MAX_STEP_LIMIT}")
        return value

    @model_validator(mode="after")
    def _state_is_consistent(self) -> Self:
        if self.step > self.max_steps:
            raise ValueError("step budget exceeded")

        expected_from = _P.INITIALIZED
        for transition in self.transitions:
            if transition.from_phase is not expected_from:
                raise ValueError("transitions must form a continuous chain from INITIALIZED")
            expected_from = transition.to_phase
        if expected_from is not self.phase:
            raise ValueError("phase must equal the last transition's target")
        steps = [
            *(record.step for record in self.transitions),
            *(record.step for record in self.directives),
            *(record.step for record in self.tool_calls),
        ]
        if any(step > self.step for step in steps):
            raise ValueError("records cannot be stamped with a future step")

        reflections = sum(1 for item in self.transitions if item.to_phase is _P.REFLECTING)
        if reflections != self.reflection.used:
            raise ValueError("reflection budget usage must match REFLECTING transitions")

        missing = self.artifacts.missing(_required_artifacts(self.phase))
        if missing:
            raise ValueError(f"phase {self.phase.value} is missing artifacts: {sorted(missing)}")

        artifact_request = self.artifacts.context.request_id if self.artifacts.context else None
        if artifact_request is not None and artifact_request != self.request_id:
            raise ValueError("artifacts belong to a different request")

        if self.phase is _P.FINALIZED:
            verification = self.artifacts.verification
            if verification is None or not verification.is_valid:
                raise ValueError("only a verified decision can be finalized")
            if self.final_decision is None or self.final_decision != self.artifacts.draft_decision:
                raise ValueError("the final decision is the verified draft")
            if self.outcome is not TerminalOutcome.FINALIZED:
                raise ValueError("FINALIZED phase requires the finalized outcome")
        elif self.phase is _P.FAILED:
            if self.outcome is not TerminalOutcome.SAFE_FALLBACK:
                raise ValueError("FAILED phase requires the safe_fallback outcome")
        elif self.outcome is not None or self.final_decision is not None:
            raise ValueError("non-terminal states carry no outcome or final decision")

        if self.final_decision is not None and self.final_decision.request_id != self.request_id:
            raise ValueError("final decision answers a different request")
        return self

    # -- queries --------------------------------------------------------------
    @property
    def is_terminal(self) -> bool:
        return self.phase in TERMINAL_PHASES

    @property
    def can_reflect(self) -> bool:
        return self.phase is _P.VERIFIED and not self.reflection.exhausted

    # -- deterministic state changes -----------------------------------------
    def _next_step(self) -> int:
        if self.step + 1 > self.max_steps:
            raise StepLimitExceededError("planner step budget exhausted")
        return self.step + 1

    def transition(self, to_phase: PlannerPhase, reason: TransitionReason) -> Self:
        """Move to ``to_phase``; raises instead of producing an invalid state."""
        if self.is_terminal:
            raise IllegalTransitionError("terminal states cannot transition")
        if to_phase not in ALLOWED_TRANSITIONS[self.phase]:
            raise IllegalTransitionError("transition not allowed from the current phase")
        if to_phase is _P.FINALIZED:
            raise IllegalTransitionError("use finalize() to complete a run")
        if to_phase is _P.FAILED:
            raise IllegalTransitionError("use fail_safe() to abort a run")
        step = self._next_step()
        reflection = self.reflection.consume() if to_phase is _P.REFLECTING else self.reflection
        artifacts = (
            self.artifacts.cleared_after(to_phase)
            if self.phase is _P.REFLECTING
            else self.artifacts
        )
        record = PhaseTransition(step=step, from_phase=self.phase, to_phase=to_phase, reason=reason)
        return self.evolve(
            phase=to_phase,
            step=step,
            reflection=reflection,
            artifacts=artifacts,
            transitions=(*self.transitions, record),
        )

    def with_artifacts(self, **updates: object) -> Self:
        return self.evolve(artifacts=self.artifacts.evolve(**updates))

    def record_directive(self, directive: PlannerDirective) -> Self:
        if self.is_terminal:
            raise IllegalTransitionError("terminal states accept no directives")
        if directive.phase is not self.phase or directive.step != self.step:
            raise IllegalTransitionError("directive must target the current phase and step")
        return self.evolve(directives=(*self.directives, directive))

    def record_tool_call(self, record: ToolCallRecord) -> Self:
        if self.is_terminal:
            raise IllegalTransitionError("terminal states accept no tool calls")
        if record.step != self.step:
            raise IllegalTransitionError("tool call must be stamped with the current step")
        if record.agent not in ROUTES[self.phase]:
            raise IllegalTransitionError("agent does not hold control in the current phase")
        return self.evolve(tool_calls=(*self.tool_calls, record))

    def finalize(self) -> Self:
        if self.phase is not _P.VERIFIED:
            raise IllegalTransitionError("only VERIFIED states can be finalized")
        step = self._next_step()
        record = PhaseTransition(
            step=step,
            from_phase=self.phase,
            to_phase=_P.FINALIZED,
            reason=TransitionReason.VERIFICATION_PASSED,
        )
        return self.evolve(
            phase=_P.FINALIZED,
            step=step,
            transitions=(*self.transitions, record),
            outcome=TerminalOutcome.FINALIZED,
            final_decision=self.artifacts.draft_decision,
        )

    def fail_safe(self, reason: TransitionReason, fallback: DecisionRecord | None = None) -> Self:
        """Abort to FAILED. ``fallback`` is a conservative decision from a deterministic tool.

        Aborting is always possible, even with the step budget spent, so a run can terminate.
        """
        if self.is_terminal:
            raise IllegalTransitionError("terminal states cannot transition")
        step = min(self.step + 1, self.max_steps)
        record = PhaseTransition(
            step=step, from_phase=self.phase, to_phase=_P.FAILED, reason=reason
        )
        return self.evolve(
            phase=_P.FAILED,
            step=step,
            transitions=(*self.transitions, record),
            outcome=TerminalOutcome.SAFE_FALLBACK,
            final_decision=fallback,
        )
