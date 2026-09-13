"""Evidence contracts: LLM agents *classify and quote*; deterministic tools *parse and verify*.

An :class:`EvidenceClaim` never carries a computed number. It carries verbatim quotes copied
from an untrusted message or image. The ``resolve_evidence`` tool checks each quote against the
source text, parses amounts/dates with :class:`~decimal.Decimal` rules, and emits
:class:`ResolvedEvidence` or :class:`RejectedClaim`. Embedded instructions are data, never
commands.
"""

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Final, Self

from pydantic import StringConstraints, model_validator

from buy_or_wait.schemas.base import StrictModel
from buy_or_wait.schemas.enums import (
    ClaimRejectionReason,
    Currency,
    EvidenceKind,
    EvidenceSource,
    ExtractorKind,
    QuoteVerification,
)
from buy_or_wait.schemas.primitives import (
    MachineKey,
    OptionalEventId,
    OptionalIsoDate,
    OptionalNonNegativeMoney,
    OptionalRequestId,
    Percentage,
    UserId,
)

_MESSAGE_ID: Final = re.compile(r"message_[0-9]+")
_IMAGE_ID: Final = re.compile(r"image_[0-9]+")

SourceId = Annotated[str, StringConstraints(pattern=r"^(message|image)_[0-9]+$")]
VerbatimQuote = Annotated[str, StringConstraints(min_length=1, max_length=240)]
"""Exact substring copied from the source. Never paraphrased, never computed."""

REQUIRED_QUOTES: Final[Mapping[EvidenceKind, frozenset[str]]] = MappingProxyType(
    {
        EvidenceKind.SALARY_CHANGE: frozenset({"amount_quote", "date_quote"}),
        EvidenceKind.TEMPORARY_SALARY_CHANGE: frozenset({"amount_quote"}),
        EvidenceKind.SALARY_DATE_CHANGE: frozenset({"date_quote"}),
        EvidenceKind.FIRST_SALARY_CONFIRMED: frozenset({"amount_quote", "date_quote"}),
        EvidenceKind.BASE_SALARY_CONFIRMED: frozenset({"amount_quote"}),
        EvidenceKind.ONE_TIME_INCOME_ADJUSTMENT: frozenset({"amount_quote"}),
        EvidenceKind.INCOME_SOURCE_REDUCED: frozenset({"amount_quote"}),
        EvidenceKind.CONFIRMED_INVOICE_SETTLEMENT: frozenset({"amount_quote", "date_quote"}),
        EvidenceKind.RECURRING_EXPENSE_CHANGE: frozenset({"percentage_quote"}),
        EvidenceKind.DOCUMENT_AMOUNT: frozenset({"amount_quote"}),
    }
)


class EvidenceClaim(StrictModel):
    """Structured interpretation proposed by an evidence agent (LLM or rule)."""

    claim_id: MachineKey
    source: EvidenceSource
    source_id: SourceId
    user_id: UserId
    request_id: OptionalRequestId = None
    related_event_id: OptionalEventId = None
    kind: EvidenceKind
    amount_quote: VerbatimQuote | None = None
    date_quote: VerbatimQuote | None = None
    percentage_quote: VerbatimQuote | None = None
    embedded_instruction_detected: bool = False
    extractor: ExtractorKind

    @model_validator(mode="after")
    def _claim_is_well_formed(self) -> Self:
        pattern = _MESSAGE_ID if self.source is EvidenceSource.MESSAGE else _IMAGE_ID
        if not pattern.fullmatch(self.source_id):
            raise ValueError("source_id does not match the evidence source")
        missing = {
            name
            for name in REQUIRED_QUOTES.get(self.kind, frozenset())
            if getattr(self, name) is None
        }
        if missing:
            raise ValueError(f"{self.kind.value} claims require quotes: {sorted(missing)}")
        if self.kind is EvidenceKind.DOCUMENT_AMOUNT and self.related_event_id is None:
            raise ValueError("document_amount claims must reference the event they complete")
        return self


class ResolvedEvidence(StrictModel):
    """A claim whose quotes were verified and parsed by the deterministic resolver."""

    claim_id: MachineKey
    source: EvidenceSource
    source_id: SourceId
    kind: EvidenceKind
    related_event_id: OptionalEventId = None
    amount: OptionalNonNegativeMoney = None
    currency: Currency | None = None
    effective_date: OptionalIsoDate = None
    percentage: Percentage | None = None
    verification: QuoteVerification

    @model_validator(mode="after")
    def _verification_matches_source(self) -> Self:
        if (
            self.source is EvidenceSource.MESSAGE
            and self.verification is QuoteVerification.IMAGE_TRANSCRIPTION
        ):
            raise ValueError("message evidence is verified against its text, not transcribed")
        if (
            self.source is EvidenceSource.IMAGE
            and self.verification is QuoteVerification.VERIFIED_IN_SOURCE_TEXT
        ):
            raise ValueError("image evidence has no source text to verify against")
        if self.amount is not None and self.currency is None:
            raise ValueError("a resolved amount needs its currency")
        if self.kind is EvidenceKind.SOLICITATION_OR_SCAM and self.amount is not None:
            raise ValueError("solicitations never produce cash amounts")
        return self


class RejectedClaim(StrictModel):
    claim_id: MachineKey
    source_id: SourceId
    reason: ClaimRejectionReason
