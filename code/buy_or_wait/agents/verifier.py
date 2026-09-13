"""Verification agent: independent checks of a proposal before it may become the decision.

1. Output contract: :func:`audit_decision` (bounds, plan shape, options, spending changes).
2. Six-tier ranking: the selected plan must be the best viable, deadline-completing,
   non-excluded plan under (deadline, no changes, total paid, start date, payment count, option id).
3. Minimum balance invariant: the selected plan is re-simulated from scratch and must never
   breach the user's minimum balance.
"""

import asyncio
from datetime import timedelta

from buy_or_wait.agents.contracts import ReasonerProposal, VerifierVerdict
from buy_or_wait.engine.plans import rank_plans
from buy_or_wait.schemas.audit import DecisionAuditContext, ValidationIssue, audit_decision
from buy_or_wait.schemas.enums import IssueCode, IssueSeverity, OutputColumn, PaymentMethod
from buy_or_wait.schemas.tools import (
    DecisionVerificationOutput,
    PlanRankingInput,
    UserFinancialContext,
)


def _error(code: IssueCode, column: OutputColumn | None, detail: str) -> ValidationIssue:
    return ValidationIssue(code=code, severity=IssueSeverity.ERROR, column=column, detail=detail)


def verify_proposal(
    context: UserFinancialContext, proposal: ReasonerProposal, horizon_days: int
) -> VerifierVerdict:
    run, decision, request = proposal.run, proposal.run.decision, context.request
    audit_context = DecisionAuditContext(
        request=request,
        profile=context.profile,
        events=context.events,
        payment_options=context.payment_options,
        option_schedules=run.option_schedules,
        horizon_end=request.request_date + timedelta(days=horizon_days),
    )
    issues = list(audit_decision(decision, audit_context))

    evaluations = {item.candidate.candidate_id: item for item in run.plan_evaluations}
    ranking = rank_plans(
        PlanRankingInput(request_id=request.request_id, evaluations=run.plan_evaluations)
    )
    expected = next(
        (
            candidate_id
            for candidate_id in ranking.ranked_candidate_ids
            if candidate_id not in proposal.excluded_candidates
            and evaluations[candidate_id].completes_by_deadline
        ),
        None,
    )
    selected = proposal.selected_candidate_id
    if selected != expected:
        issues.append(
            _error(
                IssueCode.RANKING_VIOLATION,
                OutputColumn.RECOMMENDED_PAYMENT_METHOD,
                "selected plan is not the top-ranked eligible plan",
            )
        )
    if selected is None:
        if decision.recommended_payment_method is not PaymentMethod.NOT_RECOMMENDED:
            issues.append(
                _error(
                    IssueCode.ELIGIBILITY_VIOLATION,
                    OutputColumn.RECOMMENDED_PAYMENT_METHOD,
                    "a payment was recommended without a viable plan",
                )
            )
    else:
        chosen = evaluations[selected]
        candidate = chosen.candidate
        if not chosen.is_safe:
            issues.append(
                _error(
                    IssueCode.BALANCE_BELOW_MINIMUM,
                    OutputColumn.PAYMENT_PLAN,
                    "selected plan breaches the minimum balance",
                )
            )
        if (
            candidate.method is not decision.recommended_payment_method
            or candidate.plan != decision.payment_plan
            or candidate.spending_changes != decision.spending_changes_needed
        ):
            issues.append(
                _error(
                    IssueCode.ELIGIBILITY_VIOLATION,
                    OutputColumn.PAYMENT_PLAN,
                    "decision does not reproduce the selected plan",
                )
            )

    errors = tuple(issue for issue in issues if issue.severity is IssueSeverity.ERROR)
    verification = DecisionVerificationOutput(
        request_id=request.request_id, issues=tuple(issues), is_valid=not errors
    )
    return VerifierVerdict(
        request_id=request.request_id,
        attempt=proposal.attempt,
        verification=verification,
        rejected_candidate_id=selected if errors else None,
        feedback=errors,
    )


class VerificationAgent:
    def __init__(self, horizon_days: int) -> None:
        self._horizon_days = horizon_days

    async def verify(
        self, context: UserFinancialContext, proposal: ReasonerProposal
    ) -> VerifierVerdict:
        return await asyncio.to_thread(verify_proposal, context, proposal, self._horizon_days)
