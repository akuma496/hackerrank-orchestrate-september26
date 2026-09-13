"""Reasoning agent: a deterministic ReAct loop over the ledger engine tools.

Thought -> Action (deterministic tool) -> Observation (artifact digest). The reasoner consumes
only validated ledger records plus verified evidence, and every number in its proposal comes
from a tool output. Candidates rejected by the verifier are excluded from later proposals.
"""

import asyncio

from buy_or_wait.agents.contracts import PerceptionResult, ReActStep, ReasonerProposal
from buy_or_wait.engine.policy import evaluate_request
from buy_or_wait.schemas.enums import ToolName
from buy_or_wait.schemas.tools import UserFinancialContext


class ReasoningAgent:
    def __init__(self, horizon_days: int) -> None:
        self._horizon_days = horizon_days

    async def propose(
        self,
        context: UserFinancialContext,
        perception: PerceptionResult,
        attempt: int,
        excluded: frozenset[str],
    ) -> ReasonerProposal:
        run = await asyncio.to_thread(
            evaluate_request,
            context,
            perception.evidence,
            self._horizon_days,
            excluded,
        )
        trace = (
            ReActStep(
                thought="reconstruct_cash_ledger",
                action=ToolName.NORMALIZE_LEDGER,
                observation_digest=run.ledger.digest(),
            ),
            ReActStep(
                thought="detect_supported_recurrence",
                action=ToolName.DETECT_RECURRENCE,
                observation_digest=run.recurrence.digest(),
            ),
            ReActStep(
                thought="simulate_baseline_trajectory",
                action=ToolName.FORECAST_BALANCES,
                observation_digest=run.baseline_forecast.digest(),
            ),
            ReActStep(
                thought="measure_safe_lump_sum",
                action=ToolName.COMPUTE_SAFE_AMOUNT,
                observation_digest=run.safe_amount.digest(),
            ),
            ReActStep(
                thought="find_first_safe_full_date",
                action=ToolName.FIND_EARLIEST_FULL_PAYMENT_DATE,
                observation_digest=run.earliest_full_payment.digest(),
            ),
            ReActStep(
                thought="simulate_and_rank_viable_plans",
                action=ToolName.RANK_PLANS,
                observation_digest=run.ranking.digest(),
            ),
            ReActStep(
                thought="render_grounded_explanation",
                action=ToolName.RENDER_EXPLANATION,
                observation_digest=run.decision.digest(),
            ),
        )
        return ReasonerProposal(
            request_id=context.request_id,
            attempt=attempt,
            run=run,
            trace=trace,
            excluded_candidates=tuple(sorted(excluded)),
        )
