import asyncio
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from buy_or_wait.agents.contracts import MAX_PROPOSALS, ReasonerProposal, VerifierVerdict
from buy_or_wait.agents.graph import AffordabilityGraph
from buy_or_wait.agents.perception import (
    PerceptionAgent,
    parse_transcribed_amount,
    propose_message_claims,
    resolve_message_claim,
)
from buy_or_wait.agents.reasoner import ReasoningAgent
from buy_or_wait.agents.verifier import VerificationAgent, verify_proposal
from buy_or_wait.data.loader import DatasetRepository
from buy_or_wait.schemas.audit import ValidationIssue
from buy_or_wait.schemas.enums import (
    ClaimRejectionReason,
    EvidenceKind,
    IssueCode,
    IssueSeverity,
    PaymentMethod,
    PlannerPhase,
    QuoteVerification,
)
from buy_or_wait.schemas.evidence import RejectedClaim, ResolvedEvidence
from buy_or_wait.schemas.tools import DecisionVerificationOutput, UserFinancialContext
from tests.conftest import DATASET_DIR

HORIZON = 90


@pytest.fixture(scope="module")
def repository() -> DatasetRepository:
    return DatasetRepository.load(DATASET_DIR, "requests.csv")


@pytest.fixture(scope="module")
def samples() -> DatasetRepository:
    return DatasetRepository.load(DATASET_DIR, "sample_requests.csv")


class RejectingVerifier(VerificationAgent):
    """Rejects every proposal, to exercise the bounded reflection loop."""

    def __init__(self) -> None:
        super().__init__(HORIZON)
        self.calls = 0

    async def verify(
        self, context: UserFinancialContext, proposal: ReasonerProposal
    ) -> VerifierVerdict:
        self.calls += 1
        issue = ValidationIssue(
            code=IssueCode.RANKING_VIOLATION, severity=IssueSeverity.ERROR, detail="forced"
        )
        return VerifierVerdict(
            request_id=context.request_id,
            attempt=proposal.attempt,
            verification=DecisionVerificationOutput(
                request_id=context.request_id, issues=(issue,), is_valid=False
            ),
            rejected_candidate_id=proposal.selected_candidate_id,
            feedback=(issue,),
        )


class StubVision:
    def __init__(self, line: str | None) -> None:
        self.line = line

    async def transcribe_amount_line(self, image_path: Path) -> str | None:
        return self.line


def _graph(
    verifier: VerificationAgent | None = None, vision: StubVision | None = None
) -> AffordabilityGraph:
    return AffordabilityGraph(
        PerceptionAgent(DATASET_DIR, vision),
        ReasoningAgent(HORIZON),
        verifier or VerificationAgent(HORIZON),
    )


def test_verified_proposal_is_finalized_on_first_attempt(samples: DatasetRepository) -> None:
    final, planner = asyncio.run(_graph().run(samples.context_for("request_01")))
    assert planner.phase is PlannerPhase.FINALIZED
    assert final.proposals == 1
    assert not final.fallback_used
    assert final.decision.recommended_payment_method is PaymentMethod.FULL_PAYMENT
    assert final.csv_row == final.decision.to_csv_row()


def test_third_rejection_falls_back_and_never_routes_a_fourth_proposal(
    samples: DatasetRepository,
) -> None:
    verifier = RejectingVerifier()
    final, planner = asyncio.run(_graph(verifier).run(samples.context_for("request_01")))
    assert verifier.calls == MAX_PROPOSALS
    assert final.proposals == MAX_PROPOSALS
    assert planner.phase is PlannerPhase.FAILED
    assert planner.reflection.used == planner.reflection.limit == MAX_PROPOSALS - 1
    assert final.fallback_used
    assert final.decision.recommended_payment_method is PaymentMethod.NOT_RECOMMENDED
    assert final.decision.payment_plan.is_empty


def test_graph_runs_are_reproducible(samples: DatasetRepository) -> None:
    context = samples.context_for("request_19")
    first, _ = asyncio.run(_graph().run(context))
    second, _ = asyncio.run(_graph().run(context))
    assert first.decision.digest() == second.decision.digest()


def test_verifier_accepts_engine_proposals(samples: DatasetRepository) -> None:
    context = samples.context_for("request_22")
    perception = asyncio.run(PerceptionAgent(DATASET_DIR).run(context))
    proposal = asyncio.run(ReasoningAgent(HORIZON).propose(context, perception, 1, frozenset()))
    assert verify_proposal(context, proposal, HORIZON).verification.is_valid


def test_embedded_instructions_are_rejected(repository: DatasetRepository) -> None:
    message = next(
        m for m in repository.messages_by_user["user_88"] if m.message_id == "message_67"
    )
    (claim,) = propose_message_claims(message)
    assert claim.embedded_instruction_detected
    outcome = resolve_message_claim(claim, message, date(2030, 1, 1))
    assert isinstance(outcome, RejectedClaim)
    assert outcome.reason is ClaimRejectionReason.UNTRUSTED_INSTRUCTION


def test_quotes_must_exist_verbatim_in_the_source(samples: DatasetRepository) -> None:
    message = samples.messages_by_user["user_02"][0]
    (claim,) = propose_message_claims(message)
    assert claim.kind is EvidenceKind.SALARY_CHANGE
    forged = claim.evolve(amount_quote="IDR 99999999")
    outcome = resolve_message_claim(forged, message, date(2025, 8, 5))
    assert isinstance(outcome, RejectedClaim)
    assert outcome.reason is ClaimRejectionReason.QUOTE_NOT_IN_SOURCE
    resolved = resolve_message_claim(claim, message, date(2025, 8, 5))
    assert isinstance(resolved, ResolvedEvidence)
    assert resolved.amount == Decimal("42750000.00")
    assert resolved.effective_date == date(2025, 8, 15)


def test_blank_amount_uses_worst_case_history_without_vision(samples: DatasetRepository) -> None:
    context = samples.context_for("request_19")
    perception = asyncio.run(PerceptionAgent(DATASET_DIR).run(context))
    assert perception.image_fallback_event_ids == ("event_1700",)
    (document,) = [
        r for r in perception.evidence.resolved if r.kind is EvidenceKind.DOCUMENT_AMOUNT
    ]
    history = [
        e.amount
        for e in context.events
        if e.category.value == "groceries" and e.amount is not None and e.event_id != "event_1700"
    ]
    assert document.amount == max(history)
    assert document.verification is QuoteVerification.NOT_APPLICABLE


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Total paid ₹2,298", Decimal("2298.00")),
        ("Balance Due: 1,00,000.00", Decimal("100000.00")),
        ("Net 4,365,000 and 415,800", None),
        ("no amount here", None),
    ],
)
def test_transcriptions_are_parsed_strictly(line: str, expected: Decimal | None) -> None:
    assert parse_transcribed_amount(line) == expected


def test_malformed_vision_output_falls_back(samples: DatasetRepository) -> None:
    context = samples.context_for("request_19")
    garbled = asyncio.run(PerceptionAgent(DATASET_DIR, StubVision("??")).run(context))
    assert garbled.image_fallback_event_ids == ("event_1700",)
    read = asyncio.run(PerceptionAgent(DATASET_DIR, StubVision("Item Bill ₹2854.00")).run(context))
    assert read.image_fallback_event_ids == ()
    (document,) = [r for r in read.evidence.resolved if r.kind is EvidenceKind.DOCUMENT_AMOUNT]
    assert document.amount == Decimal("2854.00")


def test_unknown_future_bill_forces_conservative_fallback(repository: DatasetRepository) -> None:
    context = repository.context_for("request_73")
    perception = asyncio.run(PerceptionAgent(DATASET_DIR).run(context))
    assert perception.unresolved_obligation_event_ids == ("event_6859",)
    final, planner = asyncio.run(_graph().run(context))
    assert planner.phase is PlannerPhase.FAILED
    assert final.proposals == 0
    assert final.decision.recommended_payment_method is PaymentMethod.NOT_RECOMMENDED
    assert final.decision.amount_safe_to_pay == Decimal(0)
