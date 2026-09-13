import decimal
from datetime import date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from buy_or_wait.schemas.decision import ScheduledPayment
from buy_or_wait.schemas.entities import ExchangeRate, FinancialProfile
from buy_or_wait.schemas.primitives import (
    decimal_policy,
    format_amount,
    format_plan_amount,
    id_ordinal,
    quantize_money,
)
from tests.conftest import Dataset


@pytest.mark.parametrize(
    "amount",
    [1.5, True, "NaN", "1e3", "1,000", " 12", "12.345", "-5", "0", None, [1]],
)
def test_positive_money_rejects_non_canonical_input(amount: object) -> None:
    with pytest.raises(ValidationError):
        ScheduledPayment.model_validate({"payment_date": "2024-01-01", "amount": amount})


@pytest.mark.parametrize(
    ("amount", "expected"),
    [("25256", Decimal("25256")), ("620.40", Decimal("620.40")), (7, Decimal(7))],
)
def test_positive_money_accepts_canonical_input(amount: object, expected: Decimal) -> None:
    payment = ScheduledPayment.model_validate({"payment_date": "2024-01-01", "amount": amount})
    assert payment.amount == expected
    assert isinstance(payment.amount, Decimal)


def test_validation_errors_never_echo_inputs() -> None:
    with pytest.raises(ValidationError) as caught:
        ScheduledPayment.model_validate({"payment_date": "2024-01-01", "amount": "-987654.32"})
    assert "987654" not in str(caught.value)


@pytest.mark.parametrize(
    "value", ["2024-3-3", "2024-02-30", "03/03/2024", "2024-03-03T00:00:00", 20240303]
)
def test_iso_date_is_strict(value: object) -> None:
    with pytest.raises(ValidationError):
        ScheduledPayment.model_validate({"payment_date": value, "amount": "1"})


def test_datetime_is_not_a_date() -> None:
    with pytest.raises(ValidationError):
        ScheduledPayment.model_validate({"payment_date": datetime(2024, 1, 1), "amount": "1"})  # noqa: DTZ001


def test_exchange_rate_direction_and_positivity() -> None:
    rate = ExchangeRate.model_validate(
        {"rate_date": "2024-04-15", "from_currency": "USD", "to_currency": "EUR", "rate": "0.92"}
    )
    assert rate.rate == Decimal("0.92")
    with pytest.raises(ValidationError):
        ExchangeRate.model_validate(
            {"rate_date": "2024-04-15", "from_currency": "USD", "to_currency": "USD", "rate": "1"}
        )
    with pytest.raises(ValidationError):
        ExchangeRate.model_validate(
            {"rate_date": "2024-04-15", "from_currency": "USD", "to_currency": "EUR", "rate": "0"}
        )


def test_float_operations_are_trapped_by_policy() -> None:
    with decimal_policy(), pytest.raises(decimal.FloatOperation):
        Decimal(1.5)  # noqa: RUF032 - deliberately mixes a float to prove the trap fires


@pytest.mark.parametrize(
    ("value", "plain", "plan"),
    [
        (Decimal("603.30"), "603.3", "603.30"),
        (Decimal("873000.00"), "873000", "873000"),
        (Decimal("620.4"), "620.4", "620.40"),
        (Decimal("15952906.67"), "15952906.67", "15952906.67"),
        (Decimal("0"), "0", "0"),
        (Decimal("23.505"), "23.5", "23.50"),
        (Decimal("23.515"), "23.52", "23.52"),
    ],
)
def test_amount_rendering(value: Decimal, plain: str, plan: str) -> None:
    assert format_amount(value) == plain
    assert format_plan_amount(value) == plan


def test_quantize_money_uses_bankers_rounding() -> None:
    assert quantize_money(Decimal("2.345")) == Decimal("2.34")
    assert quantize_money(Decimal("2.355")) == Decimal("2.36")
    assert quantize_money(Decimal("-2.345")) == Decimal("-2.34")


def test_id_ordinal_orders_numerically() -> None:
    ids = ["payment_option_100", "payment_option_9", "payment_option_10"]
    assert sorted(ids, key=id_ordinal) == [
        "payment_option_9",
        "payment_option_10",
        "payment_option_100",
    ]


def test_models_are_frozen(dataset: Dataset) -> None:
    profile = dataset.profiles["user_01"]
    with pytest.raises(ValidationError):
        profile.minimum_balance_to_keep = Decimal(0)  # type: ignore[misc]


def test_json_round_trip_is_byte_stable(dataset: Dataset) -> None:
    profile = dataset.profiles["user_03"]
    restored = FinancialProfile.model_validate_json(profile.model_dump_json())
    assert restored == profile
    assert restored.model_dump_json() == profile.model_dump_json()
    assert restored.digest() == profile.digest()


def test_evolve_revalidates(dataset: Dataset) -> None:
    payment = ScheduledPayment(payment_date=date(2024, 1, 1), amount=Decimal("10.00"))
    assert payment.evolve(amount=Decimal("12.50")).amount == Decimal("12.50")
    with pytest.raises(ValidationError):
        payment.evolve(amount=Decimal("-1"))
    with pytest.raises(ValueError, match="unknown fields"):
        payment.evolve(currency="INR")
