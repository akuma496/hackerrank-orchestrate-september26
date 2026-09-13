"""Shared fixtures: every dataset file parsed through the contracts exactly once per session."""

import csv
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Final

import pytest

from buy_or_wait.schemas.audit import DecisionAuditContext, OptionSchedule
from buy_or_wait.schemas.decision import DecisionRecord, PaymentPlan, ScheduledPayment
from buy_or_wait.schemas.entities import (
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    ImageRecord,
    Message,
    PaymentOption,
    PurchaseRequest,
)
from buy_or_wait.schemas.enums import PaymentMethod
from buy_or_wait.schemas.primitives import install_decimal_policy
from buy_or_wait.schemas.tools import DEFAULT_HORIZON_DAYS, UserFinancialContext

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATASET_DIR: Final[Path] = REPO_ROOT / "dataset"
REQUEST_COLUMNS: Final[tuple[str, ...]] = tuple(PurchaseRequest.model_fields)


def read_rows(file_name: str) -> list[dict[str, str]]:
    with (DATASET_DIR / file_name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True, slots=True)
class Dataset:
    profiles: dict[str, FinancialProfile]
    events: tuple[FinancialEvent, ...]
    rates: tuple[ExchangeRate, ...]
    requests: dict[str, PurchaseRequest]
    options: tuple[PaymentOption, ...]
    messages: tuple[Message, ...]
    images: tuple[ImageRecord, ...]
    sample_rows: tuple[dict[str, str], ...]

    def context_for(self, request_id: str) -> UserFinancialContext:
        request = self.requests[request_id]
        user_id = request.user_id
        return UserFinancialContext(
            request_id=request_id,
            request=request,
            profile=self.profiles[user_id],
            events=tuple(event for event in self.events if event.user_id == user_id),
            payment_options=tuple(
                option for option in self.options if option.request_id == request_id
            ),
            messages=tuple(message for message in self.messages if message.user_id == user_id),
            images=tuple(image for image in self.images if image.user_id == user_id),
            exchange_rates=self.rates,
        )


def schedule_for(option: PaymentOption) -> OptionSchedule:
    """Test-side reference schedule: first date plus k * frequency days."""
    step = timedelta(days=option.payment_frequency_days or 0)
    payments = tuple(
        ScheduledPayment(
            payment_date=option.first_payment_date + step * index, amount=option.payment_amount
        )
        for index in range(option.number_of_payments)
    )
    return OptionSchedule(
        payment_option_id=option.payment_option_id, plan=PaymentPlan(payments=payments)
    )


def audit_context_for(dataset: Dataset, request_id: str) -> DecisionAuditContext:
    context = dataset.context_for(request_id)
    return DecisionAuditContext(
        request=context.request,
        profile=context.profile,
        events=context.events,
        payment_options=context.payment_options,
        option_schedules=tuple(
            schedule_for(option)
            for option in context.payment_options
            if option.payment_method is PaymentMethod.INSTALLMENTS
        ),
        horizon_end=context.request.request_date + timedelta(days=DEFAULT_HORIZON_DAYS),
    )


def sample_decision(dataset: Dataset, request_id: str) -> DecisionRecord:
    row = next(row for row in dataset.sample_rows if row["request_id"] == request_id)
    return DecisionRecord.from_csv_row(row)


@pytest.fixture(scope="session", autouse=True)
def _decimal_policy() -> None:
    install_decimal_policy()


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    sample_rows = tuple(read_rows("sample_requests.csv"))
    request_rows = [*read_rows("requests.csv"), *sample_rows]
    return Dataset(
        profiles={
            row["user_id"]: FinancialProfile.model_validate(row)
            for row in read_rows("financial_profiles.csv")
        },
        events=tuple(
            FinancialEvent.model_validate(row) for row in read_rows("financial_events.csv")
        ),
        rates=tuple(ExchangeRate.model_validate(row) for row in read_rows("exchange_rates.csv")),
        requests={
            row["request_id"]: PurchaseRequest.model_validate(
                {column: row[column] for column in REQUEST_COLUMNS}
            )
            for row in request_rows
        },
        options=tuple(
            PaymentOption.model_validate(row) for row in read_rows("request_payment_options.csv")
        ),
        messages=tuple(Message.model_validate(row) for row in read_rows("messages.csv")),
        images=tuple(ImageRecord.model_validate(row) for row in read_rows("images.csv")),
        sample_rows=sample_rows,
    )
