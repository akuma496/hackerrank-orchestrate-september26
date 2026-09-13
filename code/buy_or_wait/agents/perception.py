"""Perception specialist: turns untrusted messages and images into verified evidence.

* Messages: deterministic template rules (English and Bahasa Indonesia) propose claims that carry
  verbatim quotes only. The resolver checks each quote is an exact substring of the message and
  parses amounts, dates, and percentages with strict Decimal rules.
* Images: an optional vision model transcribes one amount line. If it is unavailable, malformed,
  or unparseable, the blank amount is filled with the worst-case (maximum) historical amount of
  the same category in the same currency. Without history the amount stays unresolved and the
  ledger flags it; nothing is invented.
* Embedded instructions ("pay the release charge today") are recorded and rejected.
"""

import asyncio
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Final, Protocol

from buy_or_wait.agents.contracts import PerceptionResult
from buy_or_wait.schemas.entities import FinancialEvent, ImageRecord, Message
from buy_or_wait.schemas.enums import (
    ClaimRejectionReason,
    Currency,
    Direction,
    EventStatus,
    EvidenceKind,
    EvidenceSource,
    ExtractorKind,
    QuoteVerification,
)
from buy_or_wait.schemas.evidence import EvidenceClaim, RejectedClaim, ResolvedEvidence
from buy_or_wait.schemas.primitives import quantize_money
from buy_or_wait.schemas.tools import ResolveEvidenceOutput, UserFinancialContext

_AMOUNT: Final = r"(?:INR|ZAR|IDR|USD|EUR) ?[0-9][0-9,]*(?:\.[0-9]+)?"
_DATE: Final = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
_PERCENT: Final = r"[0-9]+(?:\.[0-9]+)?%"
_AMOUNT_PARTS: Final = re.compile(r"(INR|ZAR|IDR|USD|EUR) ?([0-9][0-9,]*(?:\.[0-9]+)?)")
_BARE_NUMBER: Final = re.compile(r"([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)")
_INSTRUCTION: Final = re.compile(
    r"(?i)release charge|processing charge|biaya pencairan|biaya pemrosesan"
)


@dataclass(frozen=True, slots=True)
class MessageRule:
    kind: EvidenceKind
    trigger: re.Pattern[str]
    amount: re.Pattern[str] | None = None
    when: re.Pattern[str] | None = None
    percent: re.Pattern[str] | None = None


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


MESSAGE_RULES: Final[tuple[MessageRule, ...]] = (
    MessageRule(
        EvidenceKind.SALARY_CHANGE,
        _rx(r"salary has increased to|Gaji bulanan Anda naik menjadi|Regular salary of"),
        _rx(_AMOUNT),
        _rx(_DATE),
    ),
    MessageRule(
        EvidenceKind.TEMPORARY_SALARY_CHANGE,
        _rx(r"temporary monthly pay is|next salary is reduced to|Gaji bulanan sementara"),
        _rx(_AMOUNT),
    ),
    MessageRule(
        EvidenceKind.SALARY_DATE_CHANGE,
        _rx(r"salary is now expected on|diperkirakan masuk pada"),
        None,
        _rx(_DATE),
    ),
    MessageRule(
        EvidenceKind.FIRST_SALARY_CONFIRMED,
        _rx(r"first salary|Gaji pertama"),
        _rx(_AMOUNT),
        _rx(_DATE),
    ),
    MessageRule(
        EvidenceKind.INCOME_SOURCE_REDUCED,
        _rx(r"remaining confirmed monthly salary is|Sisa gaji bulanan yang dikonfirmasi"),
        _rx(_AMOUNT),
    ),
    MessageRule(
        EvidenceKind.BASE_SALARY_CONFIRMED,
        _rx(r"confirmed base salary is|Gaji pokok yang dikonfirmasi"),
        _rx(_AMOUNT),
    ),
    MessageRule(
        EvidenceKind.BASE_SALARY_CONFIRMED,
        _rx(r"regular salary for the next payroll is|Gaji rutin Anda untuk penggajian"),
        _rx(_AMOUNT),
    ),
    MessageRule(
        EvidenceKind.ONE_TIME_INCOME_ADJUSTMENT,
        _rx(r"one-time arrears adjustment of|penyesuaian tunggakan satu kali sebesar"),
        _rx(r"(?:arrears adjustment of|satu kali sebesar) (" + _AMOUNT + ")"),
    ),
    MessageRule(
        EvidenceKind.INCOME_ENDED,
        _rx(
            r"employment has ended|seasonal contract has ended|Kontrak musiman saat ini "
            r"telah berakhir|Hubungan kerja Anda telah berakhir"
        ),
    ),
    MessageRule(
        EvidenceKind.CONFIRMED_INVOICE_SETTLEMENT,
        _rx(r"approved an invoice payment of|pembayaran faktur sebesar"),
        _rx(_AMOUNT),
        _rx(_DATE),
    ),
    MessageRule(
        EvidenceKind.RECURRING_EXPENSE_CHANGE,
        _rx(r"increases monthly rent by|menaikkan biaya sewa bulanan sebesar"),
        None,
        None,
        _rx(_PERCENT),
    ),
)


class VisionTranscriber(Protocol):
    async def transcribe_amount_line(self, image_path: Path) -> str | None: ...


# ---------------------------------------------------------------------------
# claim proposal (no numbers: quotes only)
# ---------------------------------------------------------------------------
def _quote(pattern: re.Pattern[str] | None, text: str) -> str | None:
    if pattern is None:
        return None
    match = pattern.search(text)
    if match is None:
        return None
    return match.group(1) if match.groups() else match.group(0)


def propose_message_claims(message: Message) -> list[EvidenceClaim]:
    text = message.message_text
    if _INSTRUCTION.search(text):
        return [
            EvidenceClaim(
                claim_id=f"{message.message_id}:solicitation",
                source=EvidenceSource.MESSAGE,
                source_id=message.message_id,
                user_id=message.user_id,
                request_id=message.request_id,
                related_event_id=message.related_event_id,
                kind=EvidenceKind.SOLICITATION_OR_SCAM,
                embedded_instruction_detected=True,
                extractor=ExtractorKind.RULE,
            )
        ]
    claims: list[EvidenceClaim] = []
    for index, rule in enumerate(MESSAGE_RULES):
        if not rule.trigger.search(text):
            continue
        quotes = {
            "amount_quote": _quote(rule.amount, text),
            "date_quote": _quote(rule.when, text),
            "percentage_quote": _quote(rule.percent, text),
        }
        needed = [
            name
            for name, pattern in (
                ("amount_quote", rule.amount),
                ("date_quote", rule.when),
                ("percentage_quote", rule.percent),
            )
            if pattern
        ]
        if any(quotes[name] is None for name in needed):
            continue  # malformed or unexpected wording: ignore rather than guess
        claims.append(
            EvidenceClaim(
                claim_id=f"{message.message_id}:rule{index:02d}",
                source=EvidenceSource.MESSAGE,
                source_id=message.message_id,
                user_id=message.user_id,
                request_id=message.request_id,
                related_event_id=message.related_event_id,
                kind=rule.kind,
                amount_quote=quotes["amount_quote"],
                date_quote=quotes["date_quote"],
                percentage_quote=quotes["percentage_quote"],
                extractor=ExtractorKind.RULE,
            )
        )
    return claims


# ---------------------------------------------------------------------------
# deterministic resolution
# ---------------------------------------------------------------------------
def parse_amount_quote(quote: str) -> tuple[Decimal, Currency] | None:
    match = _AMOUNT_PARTS.fullmatch(quote.strip())
    if match is None:
        return None
    return quantize_money(Decimal(match.group(2).replace(",", ""))), Currency(match.group(1))


def resolve_message_claim(
    claim: EvidenceClaim, message: Message | None, request_date: date
) -> ResolvedEvidence | RejectedClaim:
    def reject(reason: ClaimRejectionReason) -> RejectedClaim:
        return RejectedClaim(claim_id=claim.claim_id, source_id=claim.source_id, reason=reason)

    if message is None:
        return reject(ClaimRejectionReason.SOURCE_NOT_FOUND)
    if message.user_id != claim.user_id:
        return reject(ClaimRejectionReason.SOURCE_OWNER_MISMATCH)
    if claim.embedded_instruction_detected or claim.kind is EvidenceKind.SOLICITATION_OR_SCAM:
        return reject(ClaimRejectionReason.UNTRUSTED_INSTRUCTION)
    text = message.message_text
    for quote in (claim.amount_quote, claim.date_quote, claim.percentage_quote):
        if quote is not None and quote not in text:
            return reject(ClaimRejectionReason.QUOTE_NOT_IN_SOURCE)
    amount: Decimal | None = None
    currency: Currency | None = None
    if claim.amount_quote is not None:
        parsed = parse_amount_quote(claim.amount_quote)
        if parsed is None:
            return reject(ClaimRejectionReason.UNPARSEABLE_AMOUNT)
        amount, currency = parsed
    effective: date | None = None
    if claim.date_quote is not None:
        try:
            effective = date.fromisoformat(claim.date_quote)
        except ValueError:
            return reject(ClaimRejectionReason.UNPARSEABLE_DATE)
    percentage: Decimal | None = None
    if claim.percentage_quote is not None:
        percentage = Decimal(claim.percentage_quote.rstrip("%"))
    if message.sent_at.date() > request_date:
        return reject(ClaimRejectionReason.SOURCE_NOT_FOUND)
    return ResolvedEvidence(
        claim_id=claim.claim_id,
        source=EvidenceSource.MESSAGE,
        source_id=claim.source_id,
        kind=claim.kind,
        related_event_id=claim.related_event_id,
        amount=amount,
        currency=currency,
        effective_date=effective,
        percentage=percentage,
        verification=QuoteVerification.VERIFIED_IN_SOURCE_TEXT,
    )


def worst_case_historical_amount(
    event: FinancialEvent, events: tuple[FinancialEvent, ...]
) -> Decimal | None:
    """Maximum settled amount in the same category and currency before the blank record."""
    amounts = [
        other.amount
        for other in events
        if other.amount is not None
        and other.event_id != event.event_id
        and other.status is EventStatus.SETTLED
        and other.direction is event.direction
        and other.category is event.category
        and other.currency is event.currency
        and other.event_date <= event.event_date
    ]
    return max(amounts) if amounts else None


def unresolved_obligations(
    context: UserFinancialContext, documented_event_ids: frozenset[str]
) -> tuple[str, ...]:
    """Future debits (pending or scheduled) with a blank amount and no usable evidence."""
    return tuple(
        sorted(
            event.event_id
            for event in context.events
            if event.amount is None
            and event.direction is Direction.DEBIT
            and event.status in {EventStatus.PENDING, EventStatus.SCHEDULED}
            and event.event_id not in documented_event_ids
        )
    )


def parse_transcribed_amount(line: str) -> Decimal | None:
    numbers = _BARE_NUMBER.findall(line)
    if len(numbers) != 1:
        return None  # ambiguous or missing: never pick one
    value = Decimal(numbers[0].replace(",", ""))
    return quantize_money(value) if value > 0 else None


# ---------------------------------------------------------------------------
# agent
# ---------------------------------------------------------------------------
class PerceptionAgent:
    def __init__(self, dataset_root: Path, vision: VisionTranscriber | None = None) -> None:
        self._dataset_root = dataset_root
        self._vision = vision

    async def run(self, context: UserFinancialContext) -> PerceptionResult:
        request = context.request
        messages = sorted(context.messages, key=lambda m: m.message_id)
        message_claims = await asyncio.gather(
            *(asyncio.to_thread(propose_message_claims, message) for message in messages)
        )
        image_results = await asyncio.gather(
            *(self._image_evidence(image, context) for image in context.images)
        )
        by_id = {message.message_id: message for message in messages}
        claims = [claim for batch in message_claims for claim in batch]
        resolved: list[ResolvedEvidence] = []
        rejected: list[RejectedClaim] = []
        for claim in sorted(claims, key=lambda c: c.claim_id):
            outcome = resolve_message_claim(claim, by_id.get(claim.source_id), request.request_date)
            if isinstance(outcome, ResolvedEvidence):
                resolved.append(outcome)
            else:
                rejected.append(outcome)
        fallback_ids: list[str] = []
        for evidence, used_fallback in image_results:
            if evidence is None:
                continue
            resolved.append(evidence)
            if used_fallback and evidence.related_event_id is not None:
                fallback_ids.append(evidence.related_event_id)
        documented = frozenset(
            r.related_event_id
            for r in resolved
            if r.kind is EvidenceKind.DOCUMENT_AMOUNT and r.related_event_id is not None
        )
        ignored = tuple(
            sorted(
                m.message_id for m, batch in zip(messages, message_claims, strict=True) if not batch
            )
        )
        return PerceptionResult(
            request_id=request.request_id,
            claims=tuple(sorted(claims, key=lambda c: c.claim_id)),
            evidence=ResolveEvidenceOutput(
                request_id=request.request_id,
                resolved=tuple(sorted(resolved, key=lambda r: r.claim_id)),
                rejected=tuple(sorted(rejected, key=lambda r: r.claim_id)),
            ),
            ignored_message_ids=ignored,
            image_fallback_event_ids=tuple(sorted(fallback_ids)),
            unresolved_obligation_event_ids=unresolved_obligations(context, documented),
        )

    async def _image_evidence(
        self, image: ImageRecord, context: UserFinancialContext
    ) -> tuple[ResolvedEvidence | None, bool]:
        events = {event.event_id: event for event in context.events}
        event = events.get(image.related_event_id or "")
        if event is None or event.amount is not None or event.direction is Direction.NON_CASH:
            return None, False
        amount: Decimal | None = None
        path = self._dataset_root / image.relative_path
        if self._vision is not None and path.is_file():
            try:
                line = await self._vision.transcribe_amount_line(path)
            except Exception:
                line = None
            amount = parse_transcribed_amount(line) if line else None
        if amount is not None:
            return self._document(
                image, event, amount, QuoteVerification.IMAGE_TRANSCRIPTION, "vision"
            ), False
        fallback = worst_case_historical_amount(event, context.events)
        if fallback is None:
            return None, False
        return self._document(
            image, event, fallback, QuoteVerification.NOT_APPLICABLE, "worst_case"
        ), True

    @staticmethod
    def _document(
        image: ImageRecord,
        event: FinancialEvent,
        amount: Decimal,
        verification: QuoteVerification,
        method: str,
    ) -> ResolvedEvidence:
        return ResolvedEvidence(
            claim_id=f"{image.image_id}:{method}",
            source=EvidenceSource.IMAGE,
            source_id=image.image_id,
            kind=EvidenceKind.DOCUMENT_AMOUNT,
            related_event_id=event.event_id,
            amount=amount,
            currency=event.currency,
            verification=verification,
        )
