"""LangGraph execution graph for one affordability request.

    START -> planner -> perception -> reasoner -> verifier --route()--> output -> END
                                         ^            |
                                         +- reflect <-+  (proposal 2 and 3 only)
                                                      +-> fallback -> output   (3rd rejection)

Every phase change goes through :class:`PlannerState` (``transition`` / ``finalize`` /
``fail_safe``), which enforces legal routes, tool permissions, artifact completeness, and the
reflection budget. The only branching point is :func:`route_after_verification`, a pure function
of the state. A fourth proposal is structurally impossible: the reflection budget is two.
"""

from typing import Final, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from buy_or_wait.agents.contracts import (
    MAX_PROPOSALS,
    FinalDecision,
    PerceptionResult,
    ReasonerProposal,
    VerifierVerdict,
)
from buy_or_wait.agents.fallback import FallbackReason, fallback_decision
from buy_or_wait.agents.perception import PerceptionAgent
from buy_or_wait.agents.reasoner import ReasoningAgent
from buy_or_wait.agents.verifier import VerificationAgent
from buy_or_wait.observability.structured_logging import (
    LogLevel,
    Stopwatch,
    get_logger,
    log_event,
    request_context,
)
from buy_or_wait.schemas.base import StrictModel
from buy_or_wait.schemas.enums import (
    AgentRole,
    PlannerPhase,
    RoutingReason,
    ToolCallStatus,
    ToolName,
    TransitionReason,
)
from buy_or_wait.schemas.state import (
    PlannerDirective,
    PlannerState,
    ReflectionBudget,
    ToolCallRecord,
)
from buy_or_wait.schemas.tools import UserFinancialContext

REFLECTION_LIMIT: Final[int] = MAX_PROPOSALS - 1
RECURSION_LIMIT: Final[int] = 40
_LOG = get_logger("agents.graph")

Route = Literal["output", "reflect", "fallback"]
PerceptionRoute = Literal["reasoner", "fallback"]


class GraphState(TypedDict, total=False):
    context: UserFinancialContext
    planner: PlannerState
    perception: PerceptionResult
    proposal: ReasonerProposal
    verdict: VerifierVerdict
    excluded: tuple[str, ...]
    final: FinalDecision
    fallback_reason: FallbackReason


def _call(
    state: PlannerState, agent: AgentRole, tool: ToolName, source: StrictModel, product: StrictModel
) -> PlannerState:
    record = ToolCallRecord(
        call_id=f"s{state.step:03d}-{tool.value}",
        step=state.step,
        agent=agent,
        tool=tool,
        status=ToolCallStatus.SUCCEEDED,
        input_digest=source.digest(),
        output_digest=product.digest(),
    )
    return state.record_tool_call(record)


def _direct(
    state: PlannerState, agent: AgentRole, tool: ToolName | None, reason: RoutingReason
) -> PlannerState:
    return state.record_directive(
        PlannerDirective(
            step=state.step, phase=state.phase, next_agent=agent, tool=tool, reason=reason
        )
    )


def route_after_perception(state: GraphState) -> PerceptionRoute:
    """Unknown future obligations make every forecast unsafe: go straight to the fallback."""
    if state["perception"].unresolved_obligation_event_ids:
        return "fallback"
    return "reasoner"


def route_after_verification(state: GraphState) -> Route:
    """The single, pure routing rule after verification."""
    verdict, planner = state["verdict"], state["planner"]
    if verdict.verification.is_valid:
        return "output"
    if planner.can_reflect:
        return "reflect"
    return "fallback"


class AffordabilityGraph:
    def __init__(
        self, perception: PerceptionAgent, reasoner: ReasoningAgent, verifier: VerificationAgent
    ) -> None:
        self._perception = perception
        self._reasoner = reasoner
        self._verifier = verifier
        builder: StateGraph[GraphState] = StateGraph(GraphState)
        builder.add_node("planner", self._planner)
        builder.add_node("perception", self._perceive)
        builder.add_node("reasoner", self._reason)
        builder.add_node("verifier", self._verify)
        builder.add_node("reflect", self._reflect)
        builder.add_node("fallback", self._fallback)
        builder.add_node("output", self._output)
        builder.add_edge(START, "planner")
        builder.add_edge("planner", "perception")
        builder.add_conditional_edges(
            "perception",
            route_after_perception,
            {"reasoner": "reasoner", "fallback": "fallback"},
        )
        builder.add_edge("reasoner", "verifier")
        builder.add_conditional_edges(
            "verifier",
            route_after_verification,
            {"output": "output", "reflect": "reflect", "fallback": "fallback"},
        )
        builder.add_edge("reflect", "reasoner")
        builder.add_edge("fallback", "output")
        builder.add_edge("output", END)
        self._graph = builder.compile()

    async def run(self, context: UserFinancialContext) -> tuple[FinalDecision, PlannerState]:
        with request_context(context.request_id):
            stopwatch = Stopwatch()
            result = await self._graph.ainvoke(
                {"context": context, "excluded": ()}, config={"recursion_limit": RECURSION_LIMIT}
            )
            final: FinalDecision = result["final"]
            planner: PlannerState = result["planner"]
            log_event(
                _LOG,
                "graph.completed",
                fields={
                    "phase": planner.phase.value,
                    "status": "fallback" if final.fallback_used else "verified",
                    "attempt": final.proposals,
                    "reflection_used": planner.reflection.used,
                    "duration_ms": stopwatch.elapsed_ms(),
                    "output_digest": final.decision.digest(),
                },
            )
            return final, planner

    # -- planner: initialise state, hand off to perception -----------------------------
    async def _planner(self, state: GraphState) -> GraphState:
        context = state["context"]
        planner = PlannerState(
            request_id=context.request_id, reflection=ReflectionBudget(limit=REFLECTION_LIMIT)
        )
        planner = _direct(
            planner, AgentRole.CONTEXT_LOADER, ToolName.LOAD_USER_CONTEXT, RoutingReason.NEXT_STAGE
        )
        planner = _call(
            planner, AgentRole.CONTEXT_LOADER, ToolName.LOAD_USER_CONTEXT, context.request, context
        )
        planner = planner.with_artifacts(context=context).transition(
            PlannerPhase.CONTEXT_LOADED, TransitionReason.STAGE_COMPLETED
        )
        log_event(
            _LOG,
            "agent.completed",
            fields={
                "agent": AgentRole.PLANNER.value,
                "phase": planner.phase.value,
                "step": planner.step,
            },
        )
        return {"planner": planner}

    # -- perception: concurrent message and image ingestion ----------------------------
    async def _perceive(self, state: GraphState) -> GraphState:
        context, planner = state["context"], state["planner"]
        perception = await self._perception.run(context)
        planner = _direct(
            planner,
            AgentRole.EVIDENCE_INTERPRETER,
            ToolName.RESOLVE_EVIDENCE,
            RoutingReason.NEXT_STAGE,
        )
        planner = _call(
            planner,
            AgentRole.EVIDENCE_INTERPRETER,
            ToolName.RESOLVE_EVIDENCE,
            context,
            perception.evidence,
        )
        planner = planner.with_artifacts(evidence=perception.evidence).transition(
            PlannerPhase.EVIDENCE_RESOLVED, TransitionReason.STAGE_COMPLETED
        )
        log_event(
            _LOG,
            "agent.completed",
            fields={
                "agent": AgentRole.EVIDENCE_INTERPRETER.value,
                "phase": planner.phase.value,
                "record_count": len(perception.evidence.resolved),
            },
        )
        return {"planner": planner, "perception": perception}

    # -- reasoner: ReAct over deterministic tools --------------------------------------
    async def _reason(self, state: GraphState) -> GraphState:
        context, planner = state["context"], state["planner"]
        excluded = frozenset(state.get("excluded", ()))
        attempt = (
            sum(1 for t in planner.transitions if t.to_phase is PlannerPhase.DECISION_DRAFTED) + 1
        )
        proposal = await self._reasoner.propose(context, state["perception"], attempt, excluded)
        run = proposal.run
        if planner.phase is PlannerPhase.REFLECTING:
            planner = _direct(
                planner,
                AgentRole.PLAN_STRATEGIST,
                ToolName.RANK_PLANS,
                RoutingReason.RETRY_AFTER_VERIFICATION,
            )
            planner = planner.transition(
                PlannerPhase.PLANS_EVALUATED, TransitionReason.REFLECTION_RETRY
            )
        else:
            evidence = planner.artifacts.evidence
            if evidence is None:
                raise RuntimeError("EVIDENCE_RESOLVED state is missing its evidence artifact")
            planner = _call(
                planner, AgentRole.LEDGER_ANALYST, ToolName.NORMALIZE_LEDGER, evidence, run.ledger
            )
            planner = planner.with_artifacts(
                ledger=run.ledger, recurrence=run.recurrence
            ).transition(PlannerPhase.LEDGER_NORMALIZED, TransitionReason.STAGE_COMPLETED)
            planner = _call(
                planner,
                AgentRole.FORECASTER,
                ToolName.FORECAST_BALANCES,
                run.recurrence,
                run.baseline_forecast,
            )
            planner = planner.with_artifacts(
                baseline_forecast=run.baseline_forecast,
                safe_amount=run.safe_amount,
                earliest_full_payment=run.earliest_full_payment,
            ).transition(PlannerPhase.FORECAST_READY, TransitionReason.STAGE_COMPLETED)
            planner = _call(
                planner,
                AgentRole.PLAN_STRATEGIST,
                ToolName.RANK_PLANS,
                run.baseline_forecast,
                run.ranking,
            )
            planner = planner.with_artifacts(
                plan_evaluations=run.plan_evaluations, ranking=run.ranking
            ).transition(PlannerPhase.PLANS_EVALUATED, TransitionReason.STAGE_COMPLETED)
        planner = planner.with_artifacts(plan_evaluations=run.plan_evaluations, ranking=run.ranking)
        planner = _call(
            planner, AgentRole.EXPLAINER, ToolName.RENDER_EXPLANATION, run.ranking, run.decision
        )
        planner = planner.with_artifacts(draft_decision=run.decision).transition(
            PlannerPhase.DECISION_DRAFTED, TransitionReason.STAGE_COMPLETED
        )
        log_event(
            _LOG,
            "agent.completed",
            fields={
                "agent": "reasoner",
                "phase": planner.phase.value,
                "attempt": attempt,
                "output_digest": run.decision.digest(),
            },
        )
        return {"planner": planner, "proposal": proposal}

    # -- verifier ----------------------------------------------------------------------
    async def _verify(self, state: GraphState) -> GraphState:
        context, planner, proposal = state["context"], state["planner"], state["proposal"]
        verdict = await self._verifier.verify(context, proposal)
        planner = _call(
            planner,
            AgentRole.VERIFIER,
            ToolName.VERIFY_DECISION,
            proposal.run.decision,
            verdict.verification,
        )
        planner = planner.with_artifacts(verification=verdict.verification).transition(
            PlannerPhase.VERIFIED,
            TransitionReason.VERIFICATION_PASSED
            if verdict.verification.is_valid
            else TransitionReason.VERIFICATION_FAILED,
        )
        excluded = tuple(state.get("excluded", ()))
        if verdict.rejected_candidate_id is not None:
            excluded = tuple(sorted({*excluded, verdict.rejected_candidate_id}))
        log_event(
            _LOG,
            "agent.completed",
            fields={
                "agent": AgentRole.VERIFIER.value,
                "phase": planner.phase.value,
                "attempt": verdict.attempt,
                "status": "accepted" if verdict.verification.is_valid else "rejected",
                "issue_codes": sorted({issue.code.value for issue in verdict.feedback}),
            },
        )
        return {"planner": planner, "verdict": verdict, "excluded": excluded}

    async def _reflect(self, state: GraphState) -> GraphState:
        planner = _direct(
            state["planner"], AgentRole.PLANNER, None, RoutingReason.RETRY_AFTER_VERIFICATION
        )
        return {
            "planner": planner.transition(
                PlannerPhase.REFLECTING, TransitionReason.VERIFICATION_FAILED
            )
        }

    async def _fallback(self, state: GraphState) -> GraphState:
        context, planner = state["context"], state["planner"]
        if planner.phase is PlannerPhase.EVIDENCE_RESOLVED:
            reason = FallbackReason.PROCESSING_ERROR
            transition = TransitionReason.UNRECOVERABLE_ERROR
            decision = fallback_decision(context, reason)
            log_event(
                _LOG,
                "fallback.missing_obligation_amount",
                level=LogLevel.WARNING,
                fields={"record_count": len(state["perception"].unresolved_obligation_event_ids)},
            )
        else:
            safe = planner.artifacts.safe_amount
            earliest = planner.artifacts.earliest_full_payment
            planner = _direct(
                planner, AgentRole.PLANNER, None, RoutingReason.ABORT_TO_SAFE_FALLBACK
            )
            reason = FallbackReason.VERIFICATION_LIMIT
            transition = TransitionReason.REFLECTION_LIMIT_REACHED
            decision = fallback_decision(
                context,
                reason,
                safe.amount_safe_to_pay if safe is not None else None,
                earliest.earliest_date if earliest is not None else None,
            )
        return {"planner": planner.fail_safe(transition, decision), "fallback_reason": reason}

    # -- output: exact CSV rendering of the terminal decision --------------------------
    async def _output(self, state: GraphState) -> GraphState:
        planner = state["planner"]
        if planner.phase is PlannerPhase.VERIFIED:
            planner = planner.finalize()
        decision = planner.final_decision
        if decision is None:
            raise RuntimeError("terminal planner state carries no final decision")
        proposals = sum(
            1 for t in planner.transitions if t.to_phase is PlannerPhase.DECISION_DRAFTED
        )
        final = FinalDecision(
            request_id=decision.request_id,
            decision=decision,
            csv_row=decision.to_csv_row(),
            fallback_used=planner.phase is PlannerPhase.FAILED,
            proposals=proposals,
        )
        return {"planner": planner, "final": final}
