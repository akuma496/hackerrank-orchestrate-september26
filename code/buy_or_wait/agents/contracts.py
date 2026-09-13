"""Interface contracts between agents. Each agent consumes one model and emits one model."""

from typing import Final, Self

from pydantic import model_validator

from buy_or_wait.engine.policy import EngineRun
from buy_or_wait.schemas.audit import ValidationIssue
from buy_or_wait.schemas.base import StrictModel
from buy_or_wait.schemas.decision import DecisionRecord
from buy_or_wait.schemas.enums import ToolName
from buy_or_wait.schemas.evidence import EvidenceClaim
from buy_or_wait.schemas.primitives import MachineKey, NonNegativeCount, RequestId, Sha256Hex
from buy_or_wait.schemas.tools import DecisionVerificationOutput, ResolveEvidenceOutput

MAX_PROPOSALS: Final[int] = 3
"""Reasoner proposals per request; the third rejection ends in not_recommended."""


class PerceptionResult(StrictModel):
    """Perception -> reasoner. Only verified, parsed evidence crosses this boundary."""

    request_id: RequestId
    claims: tuple[EvidenceClaim, ...]
    evidence: ResolveEvidenceOutput
    ignored_message_ids: tuple[str, ...] = ()
    image_fallback_event_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _same_request(self) -> Self:
        if self.evidence.request_id != self.request_id:
            raise ValueError("evidence belongs to a different request")
        return self


class ReActStep(StrictModel):
    """One Thought -> Action -> Observation cycle. Observations are digests, never amounts."""

    thought: MachineKey
    action: ToolName
    observation_digest: Sha256Hex


class ReasonerProposal(StrictModel):
    """Reasoner -> verifier: a candidate decision plus the full deterministic run behind it."""

    request_id: RequestId
    attempt: NonNegativeCount
    run: EngineRun
    trace: tuple[ReActStep, ...]
    excluded_candidates: tuple[MachineKey, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.run.decision.request_id != self.request_id:
            raise ValueError("proposal decision answers a different request")
        if not 1 <= self.attempt <= MAX_PROPOSALS:
            raise ValueError(f"attempt must be within 1..{MAX_PROPOSALS}")
        return self

    @property
    def selected_candidate_id(self) -> str | None:
        return self.run.ranking.selected_candidate_id


class VerifierVerdict(StrictModel):
    """Verifier -> planner: accept the decision, or reject with machine-readable feedback."""

    request_id: RequestId
    attempt: NonNegativeCount
    verification: DecisionVerificationOutput
    rejected_candidate_id: MachineKey | None = None
    feedback: tuple[ValidationIssue, ...] = ()

    @model_validator(mode="after")
    def _feedback_matches_verdict(self) -> Self:
        if self.verification.is_valid and self.feedback:
            raise ValueError("accepted proposals carry no feedback")
        if not self.verification.is_valid and not self.feedback:
            raise ValueError("rejections must explain themselves")
        return self


class FinalDecision(StrictModel):
    """Output agent product: the decision and its exact CSV rendering."""

    request_id: RequestId
    decision: DecisionRecord
    csv_row: dict[str, str]
    fallback_used: bool
    proposals: NonNegativeCount

    @model_validator(mode="after")
    def _row_matches_decision(self) -> Self:
        if self.csv_row != self.decision.to_csv_row():
            raise ValueError("csv_row must be the exact rendering of the decision")
        return self
