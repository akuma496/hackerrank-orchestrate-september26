"""Core financial entities, one model per dataset file.

Every model validates either a raw CSV row (all strings, blanks as ``""``) or typed Python
values. Row-level invariants that hold across the whole dataset are enforced here, so malformed
or contradictory records fail at the boundary instead of deep inside a forecast.
"""

from datetime import date
from decimal import Decimal
from pathlib import PurePosixPath
from typing import Final, Self

from pydantic import field_validator, model_validator

from buy_or_wait.schemas.base import StrictModel
from buy_or_wait.schemas.enums import (
    CASH_STATUSES,
    EVENT_TYPE_DIRECTION,
    IMMEDIATE_PAYMENT_METHODS,
    OFFERABLE_PAYMENT_METHODS,
    CategoryField,
    CategorySet,
    CurrencyField,
    Direction,
    DirectionField,
    EventStatus,
    EventStatusField,
    EventTypeField,
    Flexibility,
    FlexibilityField,
    MessageSourceTypeField,
    PaymentMethod,
    PaymentMethodField,
    PaymentMethodSet,
    PriorityList,
    RequestTypeField,
)
from buy_or_wait.schemas.primitives import (
    CsvBool,
    EventId,
    ExchangeRateValue,
    ImageId,
    IsoDate,
    MessageId,
    NonNegativeMoney,
    OptionalEventId,
    OptionalIsoDate,
    OptionalNonNegativeMoney,
    OptionalPositiveCount,
    OptionalRequestId,
    PaymentOptionId,
    PositiveCount,
    PositiveMoney,
    RequestId,
    ShortText,
    SignedMoney,
    UntrustedText,
    UserId,
    UtcTimestamp,
    decimal_policy,
)

MAX_INSTALLMENT_MONTHS: Final[int] = 120
MAX_FREQUENCY_DAYS: Final[int] = 366
IMAGE_DIRECTORY: Final[PurePosixPath] = PurePosixPath("media/images")


class FinancialProfile(StrictModel):
    """A user's balance, safety floor, priorities, and payment preferences (home currency)."""

    user_id: UserId
    home_currency: CurrencyField
    current_available_balance: SignedMoney
    minimum_balance_to_keep: NonNegativeMoney
    financial_priorities: PriorityList
    expense_categories_to_protect: CategorySet
    expense_categories_user_is_willing_to_reduce: CategorySet
    expense_categories_user_is_willing_to_stop: CategorySet
    payment_methods_user_will_consider: PaymentMethodSet
    max_installment_months: OptionalPositiveCount

    @field_validator("payment_methods_user_will_consider")
    @classmethod
    def _immediate_methods_only(cls, value: frozenset[PaymentMethod]) -> frozenset[PaymentMethod]:
        if not value:
            raise ValueError("at least one payment method must be considered")
        if not value <= IMMEDIATE_PAYMENT_METHODS:
            raise ValueError("users can only consider full_payment, partial_payment, installments")
        return value

    @field_validator("max_installment_months")
    @classmethod
    def _bounded_months(cls, value: int | None) -> int | None:
        if value is not None and value > MAX_INSTALLMENT_MONTHS:
            raise ValueError(f"max_installment_months cannot exceed {MAX_INSTALLMENT_MONTHS}")
        return value

    @model_validator(mode="after")
    def _preferences_are_consistent(self) -> Self:
        considers_installments = (
            PaymentMethod.INSTALLMENTS in self.payment_methods_user_will_consider
        )
        if considers_installments != (self.max_installment_months is not None):
            raise ValueError(
                "max_installment_months is set exactly when installments are considered"
            )
        adjustable = (
            self.expense_categories_user_is_willing_to_reduce
            | self.expense_categories_user_is_willing_to_stop
        )
        if self.expense_categories_to_protect & adjustable:
            raise ValueError("a protected category cannot also be reducible or stoppable")
        return self

    def accepts(self, method: PaymentMethod) -> bool:
        return method in self.payment_methods_user_will_consider


class FinancialEvent(StrictModel):
    """One historical, pending, scheduled, failed, cancelled, or non-cash record."""

    event_id: EventId
    user_id: UserId
    event_type: EventTypeField
    description: ShortText
    category: CategoryField
    direction: DirectionField
    amount: OptionalNonNegativeMoney
    currency: CurrencyField
    event_date: IsoDate
    settlement_date: OptionalIsoDate
    status: EventStatusField
    linked_event_id: OptionalEventId
    flexibility: FlexibilityField
    minimum_allowed_amount: OptionalNonNegativeMoney

    @model_validator(mode="after")
    def _record_is_coherent(self) -> Self:
        if EVENT_TYPE_DIRECTION[self.event_type] is not self.direction:
            raise ValueError("direction does not match event_type")
        is_non_cash = self.direction is Direction.NON_CASH
        if is_non_cash != (self.status is EventStatus.UNREALIZED):
            raise ValueError("non_cash records are exactly the unrealized ones")
        if is_non_cash != (self.settlement_date is None):
            raise ValueError("settlement_date is required for every cash record")
        if self.settlement_date is not None and self.settlement_date < self.event_date:
            raise ValueError("settlement_date cannot precede event_date")
        if self.linked_event_id == self.event_id:
            raise ValueError("an event cannot link to itself")
        if self.flexibility is not Flexibility.FIXED and self.direction is not Direction.DEBIT:
            raise ValueError("only debits can be flexible")
        if (self.minimum_allowed_amount is not None) != self.flexibility.can_reduce:
            raise ValueError("minimum_allowed_amount is set exactly for reducible events")
        if (
            self.minimum_allowed_amount is not None
            and self.amount is not None
            and self.minimum_allowed_amount > self.amount
        ):
            raise ValueError("minimum_allowed_amount cannot exceed amount")
        return self

    @property
    def is_cash_record(self) -> bool:
        """Settled, pending, or scheduled cash movement (before de-duplication rules)."""
        return self.status in CASH_STATUSES

    @property
    def requires_evidence_amount(self) -> bool:
        """Blank amount on a cash record: resolve from the linked image, never assume zero."""
        return self.amount is None and self.direction is not Direction.NON_CASH

    @property
    def cash_date(self) -> date:
        """Date the cash moves: settlement date for cash records, event date otherwise."""
        return self.settlement_date if self.settlement_date is not None else self.event_date


class ExchangeRate(StrictModel):
    """Fixed dated rate: ``1 from_currency = rate to_currency`` (direction is significant)."""

    rate_date: IsoDate
    from_currency: CurrencyField
    to_currency: CurrencyField
    rate: ExchangeRateValue

    @model_validator(mode="after")
    def _distinct_currencies(self) -> Self:
        if self.from_currency is self.to_currency:
            raise ValueError("an exchange rate needs two different currencies")
        return self


class PurchaseRequest(StrictModel):
    """One affordability question, amounts in the user's home currency."""

    request_id: RequestId
    user_id: UserId
    request_date: IsoDate
    request_type: RequestTypeField
    requested_amount: PositiveMoney
    desired_completion_date: IsoDate
    allows_partial_payment: CsvBool
    request_text: UntrustedText

    @model_validator(mode="after")
    def _deadline_not_before_request(self) -> Self:
        if self.desired_completion_date < self.request_date:
            raise ValueError("desired_completion_date cannot precede request_date")
        return self


class PaymentOption(StrictModel):
    """A seller/provider offer. Installment totals include explicit financing fees."""

    payment_option_id: PaymentOptionId
    request_id: RequestId
    payment_method: PaymentMethodField
    payment_amount: PositiveMoney
    number_of_payments: PositiveCount
    first_payment_date: IsoDate
    payment_frequency_days: OptionalPositiveCount
    financing_fee: NonNegativeMoney
    total_payable_amount: PositiveMoney

    @field_validator("payment_method")
    @classmethod
    def _offerable(cls, value: PaymentMethod) -> PaymentMethod:
        if value not in OFFERABLE_PAYMENT_METHODS:
            raise ValueError("options can only offer full_payment or installments")
        return value

    @field_validator("payment_frequency_days")
    @classmethod
    def _bounded_frequency(cls, value: int | None) -> int | None:
        if value is not None and value > MAX_FREQUENCY_DAYS:
            raise ValueError(f"payment_frequency_days cannot exceed {MAX_FREQUENCY_DAYS}")
        return value

    @model_validator(mode="after")
    def _schedule_is_coherent(self) -> Self:
        if self.payment_method is PaymentMethod.FULL_PAYMENT:
            if self.number_of_payments != 1 or self.payment_frequency_days is not None:
                raise ValueError("full_payment is a single payment with no frequency")
            if self.financing_fee != 0 or self.payment_amount != self.total_payable_amount:
                raise ValueError("full_payment carries no fee and pays the total at once")
            return self
        if self.number_of_payments < 2 or self.payment_frequency_days is None:
            raise ValueError("installments need at least two payments and a frequency")
        with decimal_policy():
            scheduled_total = self.payment_amount * Decimal(self.number_of_payments)
        if scheduled_total != self.total_payable_amount:
            raise ValueError("payment_amount x number_of_payments must equal total_payable_amount")
        return self

    @property
    def base_amount(self) -> Decimal:
        """Total payable minus the explicit financing fee (the price being financed)."""
        with decimal_policy():
            return self.total_payable_amount - self.financing_fee


class Message(StrictModel):
    """Untrusted supporting evidence: may clarify facts, never overrides rules."""

    message_id: MessageId
    user_id: UserId
    request_id: OptionalRequestId
    related_event_id: OptionalEventId
    sent_at: UtcTimestamp
    source_type: MessageSourceTypeField
    message_text: UntrustedText


class ImageRecord(StrictModel):
    """Link from an image file to a user, request, and (optionally) one financial event."""

    image_id: ImageId
    user_id: UserId
    request_id: OptionalRequestId
    related_event_id: OptionalEventId

    @property
    def relative_path(self) -> PurePosixPath:
        """Path under the dataset directory: ``media/images/<image_id>.png``."""
        return IMAGE_DIRECTORY / f"{self.image_id}.png"
