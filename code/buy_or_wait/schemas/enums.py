"""Closed vocabularies: dataset columns, decision outputs, and orchestration states.

Dataset-facing enums come with a ``*Field`` annotated alias that parses the exact CSV/JSON value
under ``strict=True``. Orchestration enums are only built in Python, so they use the plain type.
"""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Final

from pydantic import PlainSerializer, PlainValidator

from buy_or_wait.schemas.primitives import (
    enum_parser,
    enum_set_parser,
    enum_tuple_parser,
    optional_enum_parser,
    render_enum,
    render_enum_set,
    render_enum_tuple,
    render_optional_enum,
)

_ENUM_JSON = PlainSerializer(render_enum, return_type=str, when_used="json")
_OPTIONAL_ENUM_JSON = PlainSerializer(
    render_optional_enum, return_type=str | None, when_used="json"
)
_ENUM_SET_JSON = PlainSerializer(render_enum_set, return_type=list[str], when_used="json")
_ENUM_TUPLE_JSON = PlainSerializer(render_enum_tuple, return_type=list[str], when_used="json")


# ===========================================================================
# Dataset vocabularies
# ===========================================================================
class Currency(StrEnum):
    INR = "INR"
    ZAR = "ZAR"
    IDR = "IDR"
    USD = "USD"
    EUR = "EUR"


class RequestType(StrEnum):
    PURCHASE = "purchase"
    TRAVEL = "travel"
    EDUCATION = "education"
    FAMILY_TRANSFER = "family_transfer"
    DEBT_REPAYMENT = "debt_repayment"
    INVESTMENT = "investment"
    HOUSING = "housing"
    EMERGENCY_EXPENSE = "emergency_expense"
    OTHER = "other"


class EventType(StrEnum):
    EXPENSE = "expense"
    SUBSCRIPTION = "subscription"
    INCOME = "income"
    DEBT_PAYMENT = "debt_payment"
    INVESTMENT_PURCHASE = "investment_purchase"
    REFUND = "refund"
    INVESTMENT_VALUATION = "investment_valuation"
    INVESTMENT_SALE = "investment_sale"


class Direction(StrEnum):
    DEBIT = "debit"
    CREDIT = "credit"
    NON_CASH = "non_cash"


class EventStatus(StrEnum):
    SETTLED = "settled"
    PENDING = "pending"
    SCHEDULED = "scheduled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    UNREALIZED = "unrealized"


class Flexibility(StrEnum):
    FIXED = "fixed"
    REDUCIBLE = "reducible"
    STOPPABLE = "stoppable"
    REDUCIBLE_OR_STOPPABLE = "reducible_or_stoppable"

    @property
    def can_reduce(self) -> bool:
        return self in {Flexibility.REDUCIBLE, Flexibility.REDUCIBLE_OR_STOPPABLE}

    @property
    def can_stop(self) -> bool:
        return self in {Flexibility.STOPPABLE, Flexibility.REDUCIBLE_OR_STOPPABLE}


class Category(StrEnum):
    GROCERIES = "groceries"
    TRANSPORT = "transport"
    DINING = "dining"
    SALARY = "salary"
    UTILITIES = "utilities"
    RENT = "rent"
    CLOUD_STORAGE = "cloud_storage"
    SHOPPING = "shopping"
    STREAMING = "streaming"
    DEBT_REPAYMENT = "debt_repayment"
    ENTERTAINMENT = "entertainment"
    INSURANCE = "insurance"
    MUSIC_SUBSCRIPTION = "music_subscription"
    HEALTHCARE = "healthcare"
    DELIVERY_MEMBERSHIP = "delivery_membership"
    EDUCATION = "education"
    HOUSING = "housing"
    GYM = "gym"
    FAMILY_SUPPORT = "family_support"
    INVESTMENT = "investment"
    WORK_EXPENSE = "work_expense"
    WINDFALL = "windfall"


class FinancialPriority(StrEnum):
    DEBT_REPAYMENT = "debt_repayment"
    EDUCATION = "education"
    EMERGENCY_SAVINGS = "emergency_savings"
    FAMILY_SUPPORT = "family_support"
    HEALTHCARE = "healthcare"
    HOUSING = "housing"
    RETIREMENT_INVESTMENT = "retirement_investment"
    TRAVEL = "travel"


class MessageSourceType(StrEnum):
    EMPLOYER = "employer"
    SERVICE_PROVIDER = "service_provider"
    FINANCIAL_SERVICE = "financial_service"
    BANK = "bank"
    MERCHANT = "merchant"


class PaymentMethod(StrEnum):
    FULL_PAYMENT = "full_payment"
    PARTIAL_PAYMENT = "partial_payment"
    INSTALLMENTS = "installments"
    WAIT = "wait"
    NOT_RECOMMENDED = "not_recommended"


class AffordabilityStatus(StrEnum):
    AFFORDABLE_NOW = "affordable_now"
    AFFORDABLE_WITH_PLAN = "affordable_with_plan"
    AFFORDABLE_LATER = "affordable_later"
    NOT_AFFORDABLE = "not_affordable"


class SpendingAction(StrEnum):
    STOP = "stop"
    REDUCE_TO = "reduce_to"


IMMEDIATE_PAYMENT_METHODS: Final[frozenset[PaymentMethod]] = frozenset(
    {PaymentMethod.FULL_PAYMENT, PaymentMethod.PARTIAL_PAYMENT, PaymentMethod.INSTALLMENTS}
)
"""Methods a user can list in ``payment_methods_user_will_consider``."""

OFFERABLE_PAYMENT_METHODS: Final[frozenset[PaymentMethod]] = frozenset(
    {PaymentMethod.FULL_PAYMENT, PaymentMethod.INSTALLMENTS}
)
"""Methods a seller/provider can supply in ``request_payment_options.csv``."""

CASH_STATUSES: Final[frozenset[EventStatus]] = frozenset(
    {EventStatus.SETTLED, EventStatus.PENDING, EventStatus.SCHEDULED}
)

EVENT_TYPE_DIRECTION: Final[Mapping[EventType, Direction]] = MappingProxyType(
    {
        EventType.EXPENSE: Direction.DEBIT,
        EventType.SUBSCRIPTION: Direction.DEBIT,
        EventType.DEBT_PAYMENT: Direction.DEBIT,
        EventType.INVESTMENT_PURCHASE: Direction.DEBIT,
        EventType.INCOME: Direction.CREDIT,
        EventType.REFUND: Direction.CREDIT,
        EventType.INVESTMENT_SALE: Direction.CREDIT,
        EventType.INVESTMENT_VALUATION: Direction.NON_CASH,
    }
)

# Strict CSV/JSON field aliases ----------------------------------------------
CurrencyField = Annotated[
    Currency, PlainValidator(enum_parser(Currency), json_schema_input_type=Currency), _ENUM_JSON
]
OptionalCurrencyField = Annotated[
    Currency | None,
    PlainValidator(optional_enum_parser(Currency), json_schema_input_type=Currency | None),
    _OPTIONAL_ENUM_JSON,
]
RequestTypeField = Annotated[
    RequestType,
    PlainValidator(enum_parser(RequestType), json_schema_input_type=RequestType),
    _ENUM_JSON,
]
EventTypeField = Annotated[
    EventType, PlainValidator(enum_parser(EventType), json_schema_input_type=EventType), _ENUM_JSON
]
DirectionField = Annotated[
    Direction, PlainValidator(enum_parser(Direction), json_schema_input_type=Direction), _ENUM_JSON
]
EventStatusField = Annotated[
    EventStatus,
    PlainValidator(enum_parser(EventStatus), json_schema_input_type=EventStatus),
    _ENUM_JSON,
]
FlexibilityField = Annotated[
    Flexibility,
    PlainValidator(enum_parser(Flexibility), json_schema_input_type=Flexibility),
    _ENUM_JSON,
]
CategoryField = Annotated[
    Category, PlainValidator(enum_parser(Category), json_schema_input_type=Category), _ENUM_JSON
]
MessageSourceTypeField = Annotated[
    MessageSourceType,
    PlainValidator(enum_parser(MessageSourceType), json_schema_input_type=MessageSourceType),
    _ENUM_JSON,
]
PaymentMethodField = Annotated[
    PaymentMethod,
    PlainValidator(enum_parser(PaymentMethod), json_schema_input_type=PaymentMethod),
    _ENUM_JSON,
]
AffordabilityStatusField = Annotated[
    AffordabilityStatus,
    PlainValidator(enum_parser(AffordabilityStatus), json_schema_input_type=AffordabilityStatus),
    _ENUM_JSON,
]
SpendingActionField = Annotated[
    SpendingAction,
    PlainValidator(enum_parser(SpendingAction), json_schema_input_type=SpendingAction),
    _ENUM_JSON,
]
CategorySet = Annotated[
    frozenset[Category],
    PlainValidator(enum_set_parser(Category), json_schema_input_type=list[Category]),
    _ENUM_SET_JSON,
]
PaymentMethodSet = Annotated[
    frozenset[PaymentMethod],
    PlainValidator(enum_set_parser(PaymentMethod), json_schema_input_type=list[PaymentMethod]),
    _ENUM_SET_JSON,
]
PriorityList = Annotated[
    tuple[FinancialPriority, ...],
    PlainValidator(
        enum_tuple_parser(FinancialPriority), json_schema_input_type=list[FinancialPriority]
    ),
    _ENUM_TUPLE_JSON,
]
SpendingActionSet = Annotated[
    frozenset[SpendingAction],
    PlainValidator(enum_set_parser(SpendingAction), json_schema_input_type=list[SpendingAction]),
    _ENUM_SET_JSON,
]


# ===========================================================================
# Output-contract vocabularies
# ===========================================================================
class OutputColumn(StrEnum):
    REQUEST_ID = "request_id"
    AMOUNT_SAFE_TO_PAY = "amount_safe_to_pay"
    AFFORDABILITY_STATUS = "affordability_status"
    RECOMMENDED_PAYMENT_METHOD = "recommended_payment_method"
    PAYMENT_PLAN = "payment_plan"
    EARLIEST_DATE_FOR_FULL_PAYMENT = "earliest_date_for_full_payment"
    SPENDING_CHANGES_NEEDED = "spending_changes_needed"
    DECISION_EXPLANATION = "decision_explanation"


class IssueSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class IssueCode(StrEnum):
    REQUEST_MISMATCH = "request_mismatch"
    AMOUNT_OUT_OF_BOUNDS = "amount_out_of_bounds"
    STATUS_INCONSISTENT = "status_inconsistent"
    METHOD_NOT_ACCEPTED = "method_not_accepted"
    PARTIAL_NOT_ALLOWED = "partial_not_allowed"
    PLAN_SHAPE_INVALID = "plan_shape_invalid"
    PLAN_TOTAL_MISMATCH = "plan_total_mismatch"
    PLAN_BEFORE_REQUEST_DATE = "plan_before_request_date"
    PLAN_NOT_IN_OPTIONS = "plan_not_in_options"
    EARLIEST_DATE_OUT_OF_RANGE = "earliest_date_out_of_range"
    DEADLINE_MISSED = "deadline_missed"
    SPENDING_CHANGE_UNKNOWN_EVENT = "spending_change_unknown_event"
    SPENDING_CHANGE_NOT_PERMITTED = "spending_change_not_permitted"
    SPENDING_CHANGE_BELOW_MINIMUM = "spending_change_below_minimum"
    SPENDING_CHANGE_NOT_A_REDUCTION = "spending_change_not_a_reduction"
    BALANCE_BELOW_MINIMUM = "balance_below_minimum"
    EXPLANATION_UNGROUNDED = "explanation_ungrounded"
    RANKING_VIOLATION = "ranking_violation"
    ELIGIBILITY_VIOLATION = "eligibility_violation"
    PROPOSAL_LIMIT_REACHED = "proposal_limit_reached"


# ===========================================================================
# Evidence vocabularies (LLM agents classify; deterministic tools resolve numbers)
# ===========================================================================
class EvidenceSource(StrEnum):
    MESSAGE = "message"
    IMAGE = "image"


class ExtractorKind(StrEnum):
    LLM = "llm"
    RULE = "rule"


class EvidenceKind(StrEnum):
    SALARY_CHANGE = "salary_change"
    TEMPORARY_SALARY_CHANGE = "temporary_salary_change"
    SALARY_DATE_CHANGE = "salary_date_change"
    FIRST_SALARY_CONFIRMED = "first_salary_confirmed"
    BASE_SALARY_CONFIRMED = "base_salary_confirmed"
    ONE_TIME_INCOME_ADJUSTMENT = "one_time_income_adjustment"
    INCOME_ENDED = "income_ended"
    INCOME_SOURCE_REDUCED = "income_source_reduced"
    UNCONFIRMED_INCOME = "unconfirmed_income"
    CONFIRMED_INVOICE_SETTLEMENT = "confirmed_invoice_settlement"
    REFUND_PENDING = "refund_pending"
    DISPUTED_CHARGE_PENDING = "disputed_charge_pending"
    INTERNAL_TRANSFER = "internal_transfer"
    ONE_OFF_CREDIT_SETTLED = "one_off_credit_settled"
    NON_CASH_VALUATION = "non_cash_valuation"
    RECURRING_EXPENSE_CHANGE = "recurring_expense_change"
    NEW_RECURRING_EXPENSE = "new_recurring_expense"
    BILL_RETRY_SCHEDULED = "bill_retry_scheduled"
    SEPARATE_OBLIGATIONS = "separate_obligations"
    FOREIGN_CURRENCY_SETTLEMENT = "foreign_currency_settlement"
    DOCUMENT_AMOUNT = "document_amount"
    SOLICITATION_OR_SCAM = "solicitation_or_scam"
    NOT_RELEVANT = "not_relevant"


class QuoteVerification(StrEnum):
    VERIFIED_IN_SOURCE_TEXT = "verified_in_source_text"
    IMAGE_TRANSCRIPTION = "image_transcription"
    NOT_APPLICABLE = "not_applicable"


class ClaimRejectionReason(StrEnum):
    SOURCE_NOT_FOUND = "source_not_found"
    SOURCE_OWNER_MISMATCH = "source_owner_mismatch"
    QUOTE_NOT_IN_SOURCE = "quote_not_in_source"
    UNPARSEABLE_AMOUNT = "unparseable_amount"
    UNPARSEABLE_DATE = "unparseable_date"
    MISSING_REQUIRED_QUOTE = "missing_required_quote"
    UNTRUSTED_INSTRUCTION = "untrusted_instruction"


# ===========================================================================
# Ledger / forecast vocabularies
# ===========================================================================
class ExclusionReason(StrEnum):
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNREALIZED_NON_CASH = "unrealized_non_cash"
    PENDING_CREDIT = "pending_credit"
    DUPLICATE_RECORD = "duplicate_record"
    INTERNAL_TRANSFER = "internal_transfer"
    SUPERSEDED_BY_LINKED_EVENT = "superseded_by_linked_event"
    SUPERSEDED_BY_EVIDENCE = "superseded_by_evidence"
    UNRESOLVED_AMOUNT = "unresolved_amount"
    ONE_OFF_HISTORICAL = "one_off_historical"


class AmountSource(StrEnum):
    DATASET = "dataset"
    IMAGE_EVIDENCE = "image_evidence"
    MESSAGE_EVIDENCE = "message_evidence"
    FX_CONVERSION = "fx_conversion"


class Cadence(StrEnum):
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"


class AmountBasis(StrEnum):
    CONSTANT = "constant"
    LATEST = "latest"
    MAXIMUM = "maximum"
    MEAN = "mean"
    EVIDENCE = "evidence"


class FlowOrigin(StrEnum):
    RECURRING_SERIES = "recurring_series"
    VARIABLE_SPENDING = "variable_spending"
    PENDING_DEBIT = "pending_debit"
    SCHEDULED_EVENT = "scheduled_event"
    CONFIRMED_INCOME = "confirmed_income"
    EVIDENCE_ADJUSTMENT = "evidence_adjustment"
    PLAN_PAYMENT = "plan_payment"


class IntradayOrdering(StrEnum):
    DEBITS_FIRST = "debits_first"
    CREDITS_FIRST = "credits_first"


class ExplanationTemplate(StrEnum):
    PAY_IN_FULL_TODAY = "pay_in_full_today"
    CHANGES_THEN_PAY_TODAY = "changes_then_pay_today"
    PARTIAL_PAYMENT = "partial_payment"
    INSTALLMENTS = "installments"
    WAIT_THEN_PAY = "wait_then_pay"
    NOT_AFFORDABLE_BY_DEADLINE = "not_affordable_by_deadline"
    NOT_AFFORDABLE_WITHIN_HORIZON = "not_affordable_within_horizon"


# ===========================================================================
# Orchestration vocabularies
# ===========================================================================
class ToolName(StrEnum):
    LOAD_USER_CONTEXT = "load_user_context"
    RESOLVE_EVIDENCE = "resolve_evidence"
    CONVERT_CURRENCY = "convert_currency"
    NORMALIZE_LEDGER = "normalize_ledger"
    DETECT_RECURRENCE = "detect_recurrence"
    PROJECT_CASH_FLOWS = "project_cash_flows"
    FORECAST_BALANCES = "forecast_balances"
    COMPUTE_SAFE_AMOUNT = "compute_safe_amount"
    FIND_EARLIEST_FULL_PAYMENT_DATE = "find_earliest_full_payment_date"
    BUILD_PAYMENT_SCHEDULE = "build_payment_schedule"
    EVALUATE_PLAN = "evaluate_plan"
    SEARCH_SPENDING_CHANGES = "search_spending_changes"
    RANK_PLANS = "rank_plans"
    RENDER_EXPLANATION = "render_explanation"
    VERIFY_DECISION = "verify_decision"


class ToolCallStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"


class AgentRole(StrEnum):
    PLANNER = "planner"
    CONTEXT_LOADER = "context_loader"
    EVIDENCE_INTERPRETER = "evidence_interpreter"
    LEDGER_ANALYST = "ledger_analyst"
    FORECASTER = "forecaster"
    PLAN_STRATEGIST = "plan_strategist"
    EXPLAINER = "explainer"
    VERIFIER = "verifier"


class PlannerPhase(StrEnum):
    INITIALIZED = "initialized"
    CONTEXT_LOADED = "context_loaded"
    EVIDENCE_RESOLVED = "evidence_resolved"
    LEDGER_NORMALIZED = "ledger_normalized"
    FORECAST_READY = "forecast_ready"
    PLANS_EVALUATED = "plans_evaluated"
    DECISION_DRAFTED = "decision_drafted"
    VERIFIED = "verified"
    REFLECTING = "reflecting"
    FINALIZED = "finalized"
    FAILED = "failed"


class TransitionReason(StrEnum):
    STAGE_COMPLETED = "stage_completed"
    VERIFICATION_PASSED = "verification_passed"
    VERIFICATION_FAILED = "verification_failed"
    REFLECTION_RETRY = "reflection_retry"
    REFLECTION_LIMIT_REACHED = "reflection_limit_reached"
    STEP_LIMIT_REACHED = "step_limit_reached"
    UNRECOVERABLE_ERROR = "unrecoverable_error"


class RoutingReason(StrEnum):
    NEXT_STAGE = "next_stage"
    RETRY_AFTER_VERIFICATION = "retry_after_verification"
    EVIDENCE_REINTERPRETATION = "evidence_reinterpretation"
    FINALIZE = "finalize"
    ABORT_TO_SAFE_FALLBACK = "abort_to_safe_fallback"


class TerminalOutcome(StrEnum):
    FINALIZED = "finalized"
    SAFE_FALLBACK = "safe_fallback"
