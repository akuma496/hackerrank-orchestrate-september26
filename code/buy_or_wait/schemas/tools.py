"""Typed I/O contracts for the deterministic tools.

Every tool is a pure function ``input -> output``: no clocks, randomness, network, or LLM calls.
All money is :class:`~decimal.Decimal` quantized to cents by the tool that produces it; the
validators below *verify* arithmetic invariants (they never compute results for the tool).
Agents may only reach financial state through these contracts.
"""

from collections.abc import Mapping
from datetime import date, timedelta
from decimal import Decimal
from types import MappingProxyType
from typing import Final, Protocol, Self, TypeVar

from pydantic import field_validator, model_validator

from buy_or_wait.schemas.audit import OptionSchedule, ValidationIssue
from buy_or_wait.schemas.base import StrictModel
from buy_or_wait.schemas.decision import (
    DecisionRecord,
    PaymentPlan,
    SpendingChangeSet,
)
from buy_or_wait.schemas.entities import (
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    ImageRecord,
    Message,
    PaymentOption,
    PurchaseRequest,
)
from buy_or_wait.schemas.enums import (
    OFFERABLE_PAYMENT_METHODS,
    AffordabilityStatus,
    AmountBasis,
    AmountSource,
    Cadence,
    Category,
    Currency,
    Direction,
    EventStatus,
    ExclusionReason,
    ExplanationTemplate,
    Flexibility,
    FlowOrigin,
    IntradayOrdering,
    IssueSeverity,
    PaymentMethod,
    SpendingAction,
    SpendingActionSet,
    ToolName,
)
from buy_or_wait.schemas.evidence import EvidenceClaim, RejectedClaim, ResolvedEvidence
from buy_or_wait.schemas.primitives import (
    EventId,
    ExchangeRateValue,
    ExplanationText,
    IsoDate,
    MachineKey,
    NonNegativeCount,
    NonNegativeMoney,
    OptionalIsoDate,
    OptionalNonNegativeMoney,
    OptionalPaymentOptionId,
    PositiveCount,
    PositiveMoney,
    RequestId,
    ShortText,
    SignedMoney,
    decimal_policy,
    id_ordinal,
)

MIN_PAYMENT_OPTIONS: Final[int] = 2
MAX_PAYMENT_OPTIONS: Final[int] = 4
MAX_FORECAST_DAYS: Final[int] = 366
DEFAULT_HORIZON_DAYS: Final[int] = 90
MAX_SPENDING_CHANGE_PROPOSALS: Final[int] = 3


# ===========================================================================
# Envelope
# ===========================================================================
class ToolInput(StrictModel):
    """Every call is scoped to exactly one request, for tracing and reproducibility."""

    request_id: RequestId


class ToolOutput(StrictModel):
    request_id: RequestId


InputT = TypeVar("InputT", bound=ToolInput, contravariant=True)
OutputT = TypeVar("OutputT", bound=ToolOutput, covariant=True)


class DeterministicTool(Protocol[InputT, OutputT]):
    """Structural interface every tool implements. Same input, same output, no side effects."""

    @property
    def name(self) -> ToolName: ...

    def run(self, payload: InputT) -> OutputT: ...


# ===========================================================================
# load_user_context
# ===========================================================================
class LoadUserContextInput(ToolInput):
    pass


class UserFinancialContext(ToolOutput):
    """All dataset records relevant to one request, cross-checked for referential integrity."""

    request: PurchaseRequest
    profile: FinancialProfile
    events: tuple[FinancialEvent, ...]
    payment_options: tuple[PaymentOption, ...]
    messages: tuple[Message, ...]
    images: tuple[ImageRecord, ...]
    exchange_rates: tuple[ExchangeRate, ...]

    @model_validator(mode="after")
    def _referentially_intact(self) -> Self:
        request, user_id = self.request, self.request.user_id
        if request.request_id != self.request_id:
            raise ValueError("context request does not match request_id")
        if self.profile.user_id != user_id:
            raise ValueError("profile belongs to a different user")
        owners = [
            *(event.user_id for event in self.events),
            *(message.user_id for message in self.messages),
            *(image.user_id for image in self.images),
        ]
        if any(owner != user_id for owner in owners):
            raise ValueError("events, messages, and images must belong to the requesting user")
        if not MIN_PAYMENT_OPTIONS <= len(self.payment_options) <= MAX_PAYMENT_OPTIONS:
            raise ValueError("a request has between two and four payment options")
        if any(option.request_id != request.request_id for option in self.payment_options):
            raise ValueError("payment options must belong to the request")
        option_ids = [option.payment_option_id for option in self.payment_options]
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("payment_option_id values must be unique")
        full_payments = [
            option
            for option in self.payment_options
            if option.payment_method is PaymentMethod.FULL_PAYMENT
        ]
        if len(full_payments) != 1:
            raise ValueError("exactly one full_payment option is expected")
        if any(option.base_amount != request.requested_amount for option in self.payment_options):
            raise ValueError("option totals minus fees must equal requested_amount")
        events_by_id = {event.event_id: event for event in self.events}
        if len(events_by_id) != len(self.events):
            raise ValueError("event_id values must be unique")
        for event in self.events:
            if event.linked_event_id is None:
                continue
            linked = events_by_id.get(event.linked_event_id)
            if linked is None or linked.event_date > event.event_date:
                raise ValueError("linked_event_id must reference an earlier event of the user")
        evidence_links = [
            *((message.related_event_id, message.request_id) for message in self.messages),
            *((image.related_event_id, image.request_id) for image in self.images),
        ]
        for related_event_id, evidence_request_id in evidence_links:
            if related_event_id is not None and related_event_id not in events_by_id:
                raise ValueError("evidence references an unknown event")
            if evidence_request_id is not None and evidence_request_id != request.request_id:
                raise ValueError("evidence references a different request")
        return self


# ===========================================================================
# resolve_evidence
# ===========================================================================
class ResolveEvidenceInput(ToolInput):
    claims: tuple[EvidenceClaim, ...]
    messages: tuple[Message, ...]
    images: tuple[ImageRecord, ...]
    events: tuple[FinancialEvent, ...]


class ResolveEvidenceOutput(ToolOutput):
    resolved: tuple[ResolvedEvidence, ...]
    rejected: tuple[RejectedClaim, ...]

    @model_validator(mode="after")
    def _each_claim_once(self) -> Self:
        claim_ids = [
            *(item.claim_id for item in self.resolved),
            *(item.claim_id for item in self.rejected),
        ]
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("every claim is resolved or rejected exactly once")
        return self


# ===========================================================================
# convert_currency
# ===========================================================================
class CurrencyConversionInput(ToolInput):
    amount: NonNegativeMoney
    source_currency: Currency
    target_currency: Currency
    settlement_date: IsoDate


class CurrencyConversionOutput(ToolOutput):
    """``converted_amount = quantize(amount x rate)`` using the stated direction only."""

    source_amount: NonNegativeMoney
    source_currency: Currency
    target_currency: Currency
    rate_date: IsoDate
    rate_applied: ExchangeRateValue
    converted_amount: NonNegativeMoney

    @model_validator(mode="after")
    def _identity_when_same_currency(self) -> Self:
        if self.source_currency is self.target_currency and (
            self.rate_applied != Decimal(1) or self.converted_amount != self.source_amount
        ):
            raise ValueError("same-currency conversion must be the identity")
        return self


# ===========================================================================
# normalize_ledger
# ===========================================================================
class LedgerEntry(StrictModel):
    """A cash movement that counts, expressed in home currency."""

    event_id: EventId
    cash_date: IsoDate
    direction: Direction
    amount: NonNegativeMoney
    status: EventStatus
    category: Category
    description: ShortText
    flexibility: Flexibility
    minimum_allowed_amount: OptionalNonNegativeMoney = None
    amount_source: AmountSource

    @model_validator(mode="after")
    def _counts_as_cash(self) -> Self:
        if self.direction is Direction.NON_CASH:
            raise ValueError("non-cash records never enter the ledger")
        if self.status not in {EventStatus.SETTLED, EventStatus.PENDING, EventStatus.SCHEDULED}:
            raise ValueError("only settled, pending, or scheduled records enter the ledger")
        if self.status is EventStatus.PENDING and self.direction is Direction.CREDIT:
            raise ValueError("pending credits are not counted until they settle")
        return self


class LedgerExclusion(StrictModel):
    event_id: EventId
    reason: ExclusionReason


class NormalizeLedgerInput(ToolInput):
    request_date: IsoDate
    home_currency: Currency
    events: tuple[FinancialEvent, ...]
    resolved_evidence: tuple[ResolvedEvidence, ...]
    exchange_rates: tuple[ExchangeRate, ...]


class NormalizeLedgerOutput(ToolOutput):
    entries: tuple[LedgerEntry, ...]
    exclusions: tuple[LedgerExclusion, ...]

    @model_validator(mode="after")
    def _partition_is_disjoint(self) -> Self:
        included = [entry.event_id for entry in self.entries]
        excluded = [item.event_id for item in self.exclusions]
        if len(set(included)) != len(included) or len(set(excluded)) != len(excluded):
            raise ValueError("an event appears at most once per partition")
        if set(included) & set(excluded):
            raise ValueError("an event is either counted or excluded, never both")
        return self


# ===========================================================================
# detect_recurrence
# ===========================================================================
class RecurringSeries(StrictModel):
    """A pattern supported by history (two or more occurrences), projected forward."""

    series_key: MachineKey
    category: Category
    description: ShortText
    direction: Direction
    cadence: Cadence
    anchor_day_of_month: int | None = None
    interval_days: PositiveCount | None = None
    projected_amount: PositiveMoney
    amount_basis: AmountBasis
    member_event_ids: tuple[EventId, ...]
    latest_event_id: EventId
    first_seen: IsoDate
    last_seen: IsoDate
    flexibility: Flexibility
    minimum_allowed_amount: OptionalNonNegativeMoney = None
    is_protected: bool
    is_essential: bool

    @model_validator(mode="after")
    def _supported_by_history(self) -> Self:
        if self.direction is Direction.NON_CASH:
            raise ValueError("series are cash movements")
        if len(self.member_event_ids) < 2 or len(set(self.member_event_ids)) != len(
            self.member_event_ids
        ):
            raise ValueError("recurrence needs at least two distinct occurrences")
        if self.latest_event_id not in self.member_event_ids:
            raise ValueError("latest_event_id must be one of the members")
        if self.first_seen > self.last_seen:
            raise ValueError("first_seen cannot follow last_seen")
        if self.cadence is Cadence.MONTHLY:
            if self.anchor_day_of_month is None or not 1 <= self.anchor_day_of_month <= 31:
                raise ValueError("monthly series need an anchor day within 1..31")
        elif self.interval_days is None:
            raise ValueError("weekly and biweekly series need interval_days")
        if self.direction is Direction.CREDIT and self.flexibility is not Flexibility.FIXED:
            raise ValueError("income series cannot be flexible")
        if (
            self.minimum_allowed_amount is not None
            and self.minimum_allowed_amount > self.projected_amount
        ):
            raise ValueError("minimum_allowed_amount cannot exceed projected_amount")
        if (
            self.is_protected
            and self.flexibility is not Flexibility.FIXED
            and not self.is_essential
        ):
            raise ValueError("protected series are essential")
        return self


class VariableSpendingProfile(StrictModel):
    """Conservative estimate for irregular essential spending (e.g. groceries, transport)."""

    category: Category
    window_start: IsoDate
    window_end: IsoDate
    observed_total: NonNegativeMoney
    observation_count: PositiveCount
    projected_amount_per_period: NonNegativeMoney
    period_days: PositiveCount
    last_seen: IsoDate
    amount_basis: AmountBasis
    is_essential: bool

    @model_validator(mode="after")
    def _window_is_ordered(self) -> Self:
        if self.window_start > self.window_end:
            raise ValueError("window_start cannot follow window_end")
        if not self.window_start <= self.last_seen <= self.window_end:
            raise ValueError("last_seen must fall inside the observation window")
        return self


class DetectRecurrenceInput(ToolInput):
    request_date: IsoDate
    profile: FinancialProfile
    ledger: NormalizeLedgerOutput


class DetectRecurrenceOutput(ToolOutput):
    recurring: tuple[RecurringSeries, ...]
    variable_spending: tuple[VariableSpendingProfile, ...]

    @model_validator(mode="after")
    def _unique_keys(self) -> Self:
        keys = [series.series_key for series in self.recurring]
        if len(set(keys)) != len(keys):
            raise ValueError("series_key values must be unique")
        return self


# ===========================================================================
# project_cash_flows
# ===========================================================================
class ProjectedCashFlow(StrictModel):
    flow_id: MachineKey
    flow_date: IsoDate
    direction: Direction
    amount: PositiveMoney
    origin: FlowOrigin
    category: Category | None = None
    source_event_id: EventId | None = None
    payment_option_id: OptionalPaymentOptionId = None
    is_essential: bool

    @model_validator(mode="after")
    def _cash_only(self) -> Self:
        if self.direction is Direction.NON_CASH:
            raise ValueError("projected flows move cash")
        if self.origin is FlowOrigin.PLAN_PAYMENT and self.direction is not Direction.DEBIT:
            raise ValueError("plan payments are debits")
        return self


def _check_flows_in_window(flows: tuple[ProjectedCashFlow, ...], start: date, end: date) -> None:
    if end < start:
        raise ValueError("window end cannot precede start")
    if (end - start).days > MAX_FORECAST_DAYS:
        raise ValueError(f"windows are limited to {MAX_FORECAST_DAYS} days")
    ids = [flow.flow_id for flow in flows]
    if len(set(ids)) != len(ids):
        raise ValueError("flow_id values must be unique")
    if any(not start <= flow.flow_date <= end for flow in flows):
        raise ValueError("every flow must fall inside the window")
    ordering = [(flow.flow_date, flow.flow_id) for flow in flows]
    if ordering != sorted(ordering):
        raise ValueError("flows must be sorted by (flow_date, flow_id) for determinism")


class ForecastContext(StrictModel):
    """Everything needed to rebuild a trajectory; shared by projection and change search."""

    start_date: IsoDate
    end_date: IsoDate
    home_currency: Currency
    opening_balance: SignedMoney
    minimum_balance: NonNegativeMoney
    intraday_ordering: IntradayOrdering = IntradayOrdering.DEBITS_FIRST
    ledger: NormalizeLedgerOutput
    recurrence: DetectRecurrenceOutput
    resolved_evidence: tuple[ResolvedEvidence, ...] = ()

    @model_validator(mode="after")
    def _window_is_valid(self) -> Self:
        if self.end_date < self.start_date:
            raise ValueError("end_date cannot precede start_date")
        if (self.end_date - self.start_date).days > MAX_FORECAST_DAYS:
            raise ValueError(f"windows are limited to {MAX_FORECAST_DAYS} days")
        return self


class ProjectCashFlowsInput(ToolInput):
    context: ForecastContext
    spending_changes: SpendingChangeSet = ()
    plan_payments: PaymentPlan = PaymentPlan()
    payment_option_id: OptionalPaymentOptionId = None


class ProjectCashFlowsOutput(ToolOutput):
    start_date: IsoDate
    end_date: IsoDate
    flows: tuple[ProjectedCashFlow, ...]

    @model_validator(mode="after")
    def _flows_are_valid(self) -> Self:
        _check_flows_in_window(self.flows, self.start_date, self.end_date)
        return self


# ===========================================================================
# forecast_balances
# ===========================================================================
class DailyBalance(StrictModel):
    balance_date: IsoDate
    opening_balance: SignedMoney
    total_credits: NonNegativeMoney
    total_debits: NonNegativeMoney
    intraday_low: SignedMoney
    closing_balance: SignedMoney

    @model_validator(mode="after")
    def _balances_reconcile(self) -> Self:
        with decimal_policy():
            expected_close = self.opening_balance + self.total_credits - self.total_debits
            debits_first_low = self.opening_balance - self.total_debits
        if self.closing_balance != expected_close:
            raise ValueError("closing = opening + credits - debits must hold exactly")
        if (
            not debits_first_low
            <= self.intraday_low
            <= min(self.opening_balance, self.closing_balance)
        ):
            raise ValueError("intraday_low is outside the reachable range for the day")
        return self


class ForecastBalancesInput(ToolInput):
    start_date: IsoDate
    end_date: IsoDate
    opening_balance: SignedMoney
    minimum_balance: NonNegativeMoney
    intraday_ordering: IntradayOrdering = IntradayOrdering.DEBITS_FIRST
    flows: tuple[ProjectedCashFlow, ...]

    @model_validator(mode="after")
    def _flows_are_valid(self) -> Self:
        _check_flows_in_window(self.flows, self.start_date, self.end_date)
        return self


class ForecastBalancesOutput(ToolOutput):
    start_date: IsoDate
    end_date: IsoDate
    minimum_balance: NonNegativeMoney
    days: tuple[DailyBalance, ...]
    lowest_balance: SignedMoney
    lowest_balance_date: IsoDate
    breaches_minimum: bool

    @model_validator(mode="after")
    def _series_is_continuous(self) -> Self:
        if self.end_date < self.start_date:
            raise ValueError("end_date cannot precede start_date")
        expected_days = (self.end_date - self.start_date).days + 1
        if len(self.days) != expected_days:
            raise ValueError("one DailyBalance per calendar day in [start_date, end_date]")
        for offset, day in enumerate(self.days):
            if day.balance_date != self.start_date + timedelta(days=offset):
                raise ValueError("daily balances must be contiguous and ordered")
        for previous, current in zip(self.days, self.days[1:], strict=False):
            if current.opening_balance != previous.closing_balance:
                raise ValueError("each day opens at the previous day's close")
        lowest = min(self.days, key=lambda day: (day.intraday_low, day.balance_date))
        if (lowest.intraday_low, lowest.balance_date) != (
            self.lowest_balance,
            self.lowest_balance_date,
        ):
            raise ValueError("lowest balance must be the first minimum intraday low")
        if self.breaches_minimum != (self.lowest_balance < self.minimum_balance):
            raise ValueError("breaches_minimum must reflect lowest_balance vs minimum_balance")
        return self


# ===========================================================================
# compute_safe_amount / find_earliest_full_payment_date
# ===========================================================================
class SafeAmountInput(ToolInput):
    request_date: IsoDate
    requested_amount: PositiveMoney
    baseline_forecast: ForecastBalancesOutput


class SafeAmountOutput(ToolOutput):
    """``amount_safe_to_pay = clamp(min headroom over the horizon, 0, requested_amount)``."""

    requested_amount: PositiveMoney
    minimum_headroom: SignedMoney
    binding_date: IsoDate
    amount_safe_to_pay: NonNegativeMoney

    @model_validator(mode="after")
    def _clamped(self) -> Self:
        expected = min(max(self.minimum_headroom, Decimal(0)), self.requested_amount)
        if self.amount_safe_to_pay != expected:
            raise ValueError("amount_safe_to_pay must equal clamp(headroom, 0, requested)")
        return self


class EarliestFullPaymentInput(ToolInput):
    request_date: IsoDate
    requested_amount: PositiveMoney
    baseline_forecast: ForecastBalancesOutput


class EarliestFullPaymentOutput(ToolOutput):
    search_start: IsoDate
    search_end: IsoDate
    earliest_date: OptionalIsoDate = None

    @model_validator(mode="after")
    def _within_search_window(self) -> Self:
        if self.search_end < self.search_start:
            raise ValueError("search_end cannot precede search_start")
        if self.earliest_date is not None and not (
            self.search_start <= self.earliest_date <= self.search_end
        ):
            raise ValueError("earliest_date must fall inside the search window")
        return self


# ===========================================================================
# build_payment_schedule
# ===========================================================================
class PaymentScheduleInput(ToolInput):
    option: PaymentOption


class PaymentScheduleOutput(ToolOutput):
    schedule: OptionSchedule
    payment_method: PaymentMethod
    number_of_payments: PositiveCount
    total_payable_amount: PositiveMoney

    @model_validator(mode="after")
    def _schedule_matches_option_terms(self) -> Self:
        if self.payment_method not in OFFERABLE_PAYMENT_METHODS:
            raise ValueError("schedules are built for offerable methods only")
        if self.schedule.plan.payment_count != self.number_of_payments:
            raise ValueError("schedule length must equal number_of_payments")
        if self.schedule.plan.total() != self.total_payable_amount:
            raise ValueError("schedule must add up to total_payable_amount")
        return self


# ===========================================================================
# evaluate_plan / search_spending_changes / rank_plans
# ===========================================================================
class CandidatePlan(StrictModel):
    candidate_id: MachineKey
    method: PaymentMethod
    plan: PaymentPlan
    payment_option_id: OptionalPaymentOptionId = None
    spending_changes: SpendingChangeSet = ()
    total_paid: PositiveMoney

    @model_validator(mode="after")
    def _candidate_is_coherent(self) -> Self:
        if self.method is PaymentMethod.NOT_RECOMMENDED:
            raise ValueError("not_recommended is the fallback, not a candidate")
        if self.plan.is_empty:
            raise ValueError("candidates schedule at least one payment")
        if self.plan.total() != self.total_paid:
            raise ValueError("total_paid must equal the sum of plan payments")
        if self.method is PaymentMethod.INSTALLMENTS and self.payment_option_id is None:
            raise ValueError("installment candidates must reference their payment option")
        if self.method is PaymentMethod.PARTIAL_PAYMENT and self.payment_option_id is not None:
            raise ValueError("partial payment is not a supplied option")
        return self


class SafetyViolation(StrictModel):
    violation_date: IsoDate
    projected_balance: SignedMoney
    minimum_balance: NonNegativeMoney
    shortfall: PositiveMoney

    @model_validator(mode="after")
    def _shortfall_is_exact(self) -> Self:
        with decimal_policy():
            expected = self.minimum_balance - self.projected_balance
        if self.shortfall != expected:
            raise ValueError("shortfall = minimum_balance - projected_balance")
        return self


class EvaluatedPlan(StrictModel):
    candidate: CandidatePlan
    is_safe: bool
    completes_by_deadline: bool
    lowest_balance: SignedMoney
    lowest_balance_date: IsoDate
    violations: tuple[SafetyViolation, ...]

    @model_validator(mode="after")
    def _safety_matches_violations(self) -> Self:
        if self.is_safe == bool(self.violations):
            raise ValueError("a plan is safe exactly when it has no violations")
        return self


class PlanEvaluationInput(ToolInput):
    candidate: CandidatePlan
    request_date: IsoDate
    desired_completion_date: IsoDate
    requested_amount: PositiveMoney
    forecast_with_plan: ForecastBalancesOutput


class PlanEvaluationOutput(ToolOutput):
    evaluation: EvaluatedPlan


class AdjustableExpense(StrictModel):
    """A flexible recurring debit the user permits changing (never a protected category)."""

    event_id: EventId
    series_key: MachineKey
    category: Category
    flexibility: Flexibility
    current_amount: PositiveMoney
    minimum_allowed_amount: OptionalNonNegativeMoney = None
    permitted_actions: SpendingActionSet

    @field_validator("permitted_actions")
    @classmethod
    def _non_empty(cls, value: frozenset[SpendingAction]) -> frozenset[SpendingAction]:
        if not value:
            raise ValueError("an adjustable expense permits at least one action")
        return value

    @model_validator(mode="after")
    def _actions_match_flexibility(self) -> Self:
        if SpendingAction.STOP in self.permitted_actions and not self.flexibility.can_stop:
            raise ValueError("stop requires a stoppable expense")
        if SpendingAction.REDUCE_TO in self.permitted_actions and (
            not self.flexibility.can_reduce or self.minimum_allowed_amount is None
        ):
            raise ValueError("reduce_to requires a reducible expense with a minimum amount")
        return self


class SpendingChangeSearchInput(ToolInput):
    candidate: CandidatePlan
    context: ForecastContext
    request_date: IsoDate
    desired_completion_date: IsoDate
    requested_amount: PositiveMoney
    adjustable_expenses: tuple[AdjustableExpense, ...]
    max_changes: PositiveCount = MAX_SPENDING_CHANGE_PROPOSALS

    @field_validator("max_changes")
    @classmethod
    def _at_most_three(cls, value: int) -> int:
        if value > MAX_SPENDING_CHANGE_PROPOSALS:
            raise ValueError("at most three spending changes may be proposed")
        return value


class SpendingChangeSearchOutput(ToolOutput):
    found: bool
    changes: SpendingChangeSet
    projected_savings: NonNegativeMoney

    @model_validator(mode="after")
    def _found_matches_changes(self) -> Self:
        if self.found != bool(self.changes):
            raise ValueError("found is true exactly when changes are proposed")
        if not self.found and self.projected_savings != 0:
            raise ValueError("no savings without changes")
        return self


class PlanRankKey(StrictModel):
    """Policy order from the specification, compared lexicographically (smaller is better)."""

    misses_deadline: bool
    requires_spending_changes: bool
    total_paid: PositiveMoney
    first_payment_date: IsoDate
    payment_count: PositiveCount
    payment_option_ordinal: NonNegativeCount

    def as_tuple(self) -> tuple[bool, bool, Decimal, date, int, int]:
        return (
            self.misses_deadline,
            self.requires_spending_changes,
            self.total_paid,
            self.first_payment_date,
            self.payment_count,
            self.payment_option_ordinal,
        )

    @classmethod
    def for_evaluation(cls, evaluated: EvaluatedPlan) -> Self:
        candidate = evaluated.candidate
        option_id = candidate.payment_option_id
        return cls(
            misses_deadline=not evaluated.completes_by_deadline,
            requires_spending_changes=bool(candidate.spending_changes),
            total_paid=candidate.total_paid,
            first_payment_date=candidate.plan.payments[0].payment_date,
            payment_count=candidate.plan.payment_count,
            payment_option_ordinal=0 if option_id is None else id_ordinal(option_id),
        )


class PlanRankingInput(ToolInput):
    evaluations: tuple[EvaluatedPlan, ...]


class PlanRankingOutput(ToolOutput):
    ranked_candidate_ids: tuple[MachineKey, ...]
    selected_candidate_id: MachineKey | None = None

    @model_validator(mode="after")
    def _selection_is_ranked(self) -> Self:
        if len(set(self.ranked_candidate_ids)) != len(self.ranked_candidate_ids):
            raise ValueError("ranked candidate ids must be unique")
        if (
            self.selected_candidate_id is not None
            and self.selected_candidate_id not in self.ranked_candidate_ids
        ):
            raise ValueError("the selected candidate must be ranked")
        return self


# ===========================================================================
# render_explanation / verify_decision
# ===========================================================================
class ExplanationFacts(StrictModel):
    """Grounded facts the explanation may cite; the renderer invents nothing."""

    currency: Currency
    status: AffordabilityStatus
    method: PaymentMethod
    requested_amount: PositiveMoney
    amount_safe_to_pay: NonNegativeMoney
    minimum_balance: NonNegativeMoney
    plan: PaymentPlan
    earliest_date: OptionalIsoDate = None
    desired_completion_date: IsoDate
    lowest_projected_balance: SignedMoney | None = None
    spending_change_labels: tuple[ShortText, ...] = ()
    partial_payment_available: bool = False
    horizon_days: PositiveCount = DEFAULT_HORIZON_DAYS


class ExplanationInput(ToolInput):
    facts: ExplanationFacts


class ExplanationOutput(ToolOutput):
    template: ExplanationTemplate
    text: ExplanationText


class DecisionVerificationInput(ToolInput):
    context: UserFinancialContext
    decision: DecisionRecord
    option_schedules: tuple[OptionSchedule, ...]
    horizon_end: IsoDate


class DecisionVerificationOutput(ToolOutput):
    issues: tuple[ValidationIssue, ...]
    is_valid: bool

    @model_validator(mode="after")
    def _validity_matches_issues(self) -> Self:
        has_errors = any(issue.severity is IssueSeverity.ERROR for issue in self.issues)
        if self.is_valid == has_errors:
            raise ValueError("is_valid is true exactly when no error-level issue exists")
        return self


# ===========================================================================
# Registry
# ===========================================================================
class ToolContract(StrictModel):
    name: ToolName
    input_model: type[ToolInput]
    output_model: type[ToolOutput]


TOOL_CONTRACTS: Final[Mapping[ToolName, ToolContract]] = MappingProxyType(
    {
        contract.name: contract
        for contract in (
            ToolContract(
                name=ToolName.LOAD_USER_CONTEXT,
                input_model=LoadUserContextInput,
                output_model=UserFinancialContext,
            ),
            ToolContract(
                name=ToolName.RESOLVE_EVIDENCE,
                input_model=ResolveEvidenceInput,
                output_model=ResolveEvidenceOutput,
            ),
            ToolContract(
                name=ToolName.CONVERT_CURRENCY,
                input_model=CurrencyConversionInput,
                output_model=CurrencyConversionOutput,
            ),
            ToolContract(
                name=ToolName.NORMALIZE_LEDGER,
                input_model=NormalizeLedgerInput,
                output_model=NormalizeLedgerOutput,
            ),
            ToolContract(
                name=ToolName.DETECT_RECURRENCE,
                input_model=DetectRecurrenceInput,
                output_model=DetectRecurrenceOutput,
            ),
            ToolContract(
                name=ToolName.PROJECT_CASH_FLOWS,
                input_model=ProjectCashFlowsInput,
                output_model=ProjectCashFlowsOutput,
            ),
            ToolContract(
                name=ToolName.FORECAST_BALANCES,
                input_model=ForecastBalancesInput,
                output_model=ForecastBalancesOutput,
            ),
            ToolContract(
                name=ToolName.COMPUTE_SAFE_AMOUNT,
                input_model=SafeAmountInput,
                output_model=SafeAmountOutput,
            ),
            ToolContract(
                name=ToolName.FIND_EARLIEST_FULL_PAYMENT_DATE,
                input_model=EarliestFullPaymentInput,
                output_model=EarliestFullPaymentOutput,
            ),
            ToolContract(
                name=ToolName.BUILD_PAYMENT_SCHEDULE,
                input_model=PaymentScheduleInput,
                output_model=PaymentScheduleOutput,
            ),
            ToolContract(
                name=ToolName.EVALUATE_PLAN,
                input_model=PlanEvaluationInput,
                output_model=PlanEvaluationOutput,
            ),
            ToolContract(
                name=ToolName.SEARCH_SPENDING_CHANGES,
                input_model=SpendingChangeSearchInput,
                output_model=SpendingChangeSearchOutput,
            ),
            ToolContract(
                name=ToolName.RANK_PLANS,
                input_model=PlanRankingInput,
                output_model=PlanRankingOutput,
            ),
            ToolContract(
                name=ToolName.RENDER_EXPLANATION,
                input_model=ExplanationInput,
                output_model=ExplanationOutput,
            ),
            ToolContract(
                name=ToolName.VERIFY_DECISION,
                input_model=DecisionVerificationInput,
                output_model=DecisionVerificationOutput,
            ),
        )
    }
)
