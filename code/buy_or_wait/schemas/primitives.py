"""Deterministic primitives shared by every schema: money, rates, dates, counts, identifiers.

Binary floating point is banned. Every numeric parser rejects ``float`` input, money is always
:class:`decimal.Decimal`, and :func:`decimal_policy` traps :class:`decimal.FloatOperation` so an
accidental float mix raises instead of silently rounding.

Each annotated type pairs a ``PlainValidator`` (accepts typed Python values *or* the canonical
CSV/JSON string form, nothing else) with a ``PlainSerializer`` (stable JSON rendering), so models
round-trip through JSON byte-for-byte under ``strict=True``.
"""

import decimal
import re
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Final, TypeVar

from pydantic import (
    AfterValidator,
    BeforeValidator,
    PlainSerializer,
    PlainValidator,
    StringConstraints,
)

EnumT = TypeVar("EnumT", bound=StrEnum)

DECIMAL_PRECISION: Final[int] = 34
MONEY_PLACES: Final[int] = 2
RATE_MAX_PLACES: Final[int] = 10
PERCENT_MAX_PLACES: Final[int] = 4
MONEY_QUANTUM: Final[Decimal] = Decimal("0.01")
MONEY_ROUNDING: Final[str] = decimal.ROUND_HALF_EVEN
LIST_SEPARATOR: Final[str] = "|"

_DECIMAL_TEXT: Final = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")
_COUNT_TEXT: Final = re.compile(r"0|[1-9][0-9]*")
_ISO_DATE_TEXT: Final = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_UTC_TIMESTAMP_TEXT: Final = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_ID_SUFFIX: Final = re.compile(r"_([0-9]+)$")


# ---------------------------------------------------------------------------
# Decimal policy
# ---------------------------------------------------------------------------
def build_decimal_context() -> decimal.Context:
    """Return the engine's decimal context: 34 significant digits, float mixing trapped."""
    return decimal.Context(
        prec=DECIMAL_PRECISION,
        rounding=decimal.ROUND_HALF_EVEN,
        traps=[
            decimal.FloatOperation,
            decimal.InvalidOperation,
            decimal.DivisionByZero,
            decimal.Overflow,
        ],
    )


def install_decimal_policy() -> None:
    """Install the engine decimal context for the current thread (call once at startup)."""
    decimal.setcontext(build_decimal_context())


@contextmanager
def decimal_policy() -> Iterator[decimal.Context]:
    """Scope arithmetic to the engine decimal context."""
    with decimal.localcontext(build_decimal_context()) as context:
        yield context


def quantize_money(value: Decimal) -> Decimal:
    """Round to cents with the single, explicit money rounding rule (banker's: ROUND_HALF_EVEN)."""
    with decimal_policy():
        return value.quantize(MONEY_QUANTUM, rounding=MONEY_ROUNDING)


def decimal_places(value: Decimal) -> int:
    """Number of digits after the decimal point in ``value``'s exact representation."""
    exponent = value.as_tuple().exponent
    if not isinstance(exponent, int):
        raise ValueError("decimal values must be finite")
    return max(0, -exponent)


# ---------------------------------------------------------------------------
# Parsers (strict: typed value or canonical text, never float)
# ---------------------------------------------------------------------------
def is_blank(value: object) -> bool:
    """CSV cells use the empty string for "absent"; JSON uses null."""
    return value is None or value == ""


def blank_to_none(value: object) -> object:
    return None if is_blank(value) else value


def parse_decimal(value: object) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("booleans are not valid decimal values")
    if isinstance(value, float):
        raise ValueError("binary floating point is banned; pass Decimal, int, or decimal text")
    if isinstance(value, Decimal):
        parsed = value
    elif isinstance(value, int) or (isinstance(value, str) and _DECIMAL_TEXT.fullmatch(value)):
        parsed = Decimal(value)
    else:
        raise ValueError("expected a canonical decimal such as '1234.50'")
    if not parsed.is_finite():
        raise ValueError("decimal values must be finite")
    return parsed


def parse_optional_decimal(value: object) -> Decimal | None:
    return None if is_blank(value) else parse_decimal(value)


def parse_iso_date(value: object) -> date:
    if isinstance(value, datetime):
        raise ValueError("expected a calendar date, not a datetime")
    if isinstance(value, date):
        return value
    if isinstance(value, str) and _ISO_DATE_TEXT.fullmatch(value):
        return date.fromisoformat(value)
    raise ValueError("expected an ISO-8601 calendar date 'YYYY-MM-DD'")


def parse_optional_iso_date(value: object) -> date | None:
    return None if is_blank(value) else parse_iso_date(value)


def parse_utc_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("timestamps must be timezone-aware UTC")
        return value
    if isinstance(value, str) and _UTC_TIMESTAMP_TEXT.fullmatch(value):
        return datetime.fromisoformat(value)
    raise ValueError("expected a UTC timestamp 'YYYY-MM-DDTHH:MM:SSZ'")


def parse_csv_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError("expected a boolean or the literal 'true' / 'false'")


def parse_count(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("booleans are not valid counts")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and _COUNT_TEXT.fullmatch(value):
        return int(value)
    raise ValueError("expected a non-negative integer")


def parse_optional_count(value: object) -> int | None:
    return None if is_blank(value) else parse_count(value)


def enum_parser(enum_type: type[EnumT]) -> Callable[[object], EnumT]:
    """Build a strict parser that accepts a member or its exact value (input never echoed)."""
    allowed = ", ".join(member.value for member in enum_type)

    def parse(value: object) -> EnumT:
        if isinstance(value, enum_type):
            return value
        if isinstance(value, str):
            for member in enum_type:
                if member.value == value:
                    return member
        raise ValueError(f"expected one of: {allowed}")

    return parse


def optional_enum_parser(enum_type: type[EnumT]) -> Callable[[object], EnumT | None]:
    parse_member = enum_parser(enum_type)

    def parse(value: object) -> EnumT | None:
        return None if is_blank(value) else parse_member(value)

    return parse


def _collection_items(value: object) -> tuple[object, ...]:
    if isinstance(value, str):
        return () if value == "" else tuple(value.split(LIST_SEPARATOR))
    if isinstance(value, list | tuple | frozenset):
        return tuple(value)
    raise ValueError("expected a '|'-separated string or a sequence")


def enum_set_parser(enum_type: type[EnumT]) -> Callable[[object], frozenset[EnumT]]:
    """Parse ``'a|b'`` (CSV) or a JSON array into a duplicate-free frozenset of members."""
    parse_member = enum_parser(enum_type)

    def parse(value: object) -> frozenset[EnumT]:
        members = [parse_member(item) for item in _collection_items(value)]
        if len(set(members)) != len(members):
            raise ValueError("duplicate entries are not allowed")
        return frozenset(members)

    return parse


def enum_tuple_parser(enum_type: type[EnumT]) -> Callable[[object], tuple[EnumT, ...]]:
    """Parse an *ordered*, duplicate-free list of members."""
    parse_member = enum_parser(enum_type)

    def parse(value: object) -> tuple[EnumT, ...]:
        members = tuple(parse_member(item) for item in _collection_items(value))
        if len(set(members)) != len(members):
            raise ValueError("duplicate entries are not allowed")
        return members

    return parse


# ---------------------------------------------------------------------------
# JSON renderers (deterministic; sets are sorted so digests are hash-seed independent)
# ---------------------------------------------------------------------------
def render_decimal(value: Decimal) -> str:
    return format(value, "f")


def render_optional_decimal(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def render_date(value: date) -> str:
    return value.isoformat()


def render_optional_date(value: date | None) -> str | None:
    return None if value is None else value.isoformat()


def render_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def render_int(value: int) -> int:
    return value


def render_optional_int(value: int | None) -> int | None:
    return value


def render_bool(value: bool) -> bool:
    return value


def render_enum(value: StrEnum) -> str:
    return value.value


def render_optional_enum(value: StrEnum | None) -> str | None:
    return None if value is None else value.value


def render_enum_set(values: Iterable[StrEnum]) -> list[str]:
    return sorted(member.value for member in values)


def render_enum_tuple(values: Iterable[StrEnum]) -> list[str]:
    return [member.value for member in values]


# ---------------------------------------------------------------------------
# Post-parse checks
# ---------------------------------------------------------------------------
def check_money_scale(value: Decimal) -> Decimal:
    if decimal_places(value) > MONEY_PLACES:
        raise ValueError(f"money allows at most {MONEY_PLACES} decimal places; quantize first")
    return value


def check_non_negative_money(value: Decimal) -> Decimal:
    check_money_scale(value)
    if value < 0:
        raise ValueError("amount must be non-negative")
    return value


def check_positive_money(value: Decimal) -> Decimal:
    check_money_scale(value)
    if value <= 0:
        raise ValueError("amount must be positive")
    return value


def check_optional_non_negative_money(value: Decimal | None) -> Decimal | None:
    return None if value is None else check_non_negative_money(value)


def check_optional_positive_money(value: Decimal | None) -> Decimal | None:
    return None if value is None else check_positive_money(value)


def check_rate(value: Decimal) -> Decimal:
    if decimal_places(value) > RATE_MAX_PLACES:
        raise ValueError(f"rates allow at most {RATE_MAX_PLACES} decimal places")
    if value <= 0:
        raise ValueError("rates must be positive")
    return value


def check_percentage(value: Decimal) -> Decimal:
    if decimal_places(value) > PERCENT_MAX_PLACES:
        raise ValueError(f"percentages allow at most {PERCENT_MAX_PLACES} decimal places")
    if not Decimal(0) <= value <= Decimal(1000):
        raise ValueError("percentage must be within [0, 1000]")
    return value


def check_positive_count(value: int) -> int:
    if value < 1:
        raise ValueError("count must be at least 1")
    return value


def check_optional_positive_count(value: int | None) -> int | None:
    return None if value is None else check_positive_count(value)


# ---------------------------------------------------------------------------
# Annotated types
# ---------------------------------------------------------------------------
_DECIMAL_JSON = PlainSerializer(render_decimal, return_type=str, when_used="json")
_OPTIONAL_DECIMAL_JSON = PlainSerializer(
    render_optional_decimal, return_type=str | None, when_used="json"
)

SignedMoney = Annotated[
    Decimal,
    PlainValidator(parse_decimal, json_schema_input_type=str),
    AfterValidator(check_money_scale),
    _DECIMAL_JSON,
]
"""Home-currency money that may be negative (balances, deltas); at most 2 decimal places."""

NonNegativeMoney = Annotated[
    Decimal,
    PlainValidator(parse_decimal, json_schema_input_type=str),
    AfterValidator(check_non_negative_money),
    _DECIMAL_JSON,
]

PositiveMoney = Annotated[
    Decimal,
    PlainValidator(parse_decimal, json_schema_input_type=str),
    AfterValidator(check_positive_money),
    _DECIMAL_JSON,
]

OptionalNonNegativeMoney = Annotated[
    Decimal | None,
    PlainValidator(parse_optional_decimal, json_schema_input_type=str | None),
    AfterValidator(check_optional_non_negative_money),
    _OPTIONAL_DECIMAL_JSON,
]

OptionalPositiveMoney = Annotated[
    Decimal | None,
    PlainValidator(parse_optional_decimal, json_schema_input_type=str | None),
    AfterValidator(check_optional_positive_money),
    _OPTIONAL_DECIMAL_JSON,
]

ExchangeRateValue = Annotated[
    Decimal,
    PlainValidator(parse_decimal, json_schema_input_type=str),
    AfterValidator(check_rate),
    _DECIMAL_JSON,
]

Percentage = Annotated[
    Decimal,
    PlainValidator(parse_decimal, json_schema_input_type=str),
    AfterValidator(check_percentage),
    _DECIMAL_JSON,
]

IsoDate = Annotated[
    date,
    PlainValidator(parse_iso_date, json_schema_input_type=str),
    PlainSerializer(render_date, return_type=str, when_used="json"),
]

OptionalIsoDate = Annotated[
    date | None,
    PlainValidator(parse_optional_iso_date, json_schema_input_type=str | None),
    PlainSerializer(render_optional_date, return_type=str | None, when_used="json"),
]

UtcTimestamp = Annotated[
    datetime,
    PlainValidator(parse_utc_timestamp, json_schema_input_type=str),
    PlainSerializer(render_timestamp, return_type=str, when_used="json"),
]

CsvBool = Annotated[
    bool,
    PlainValidator(parse_csv_bool, json_schema_input_type=bool),
    PlainSerializer(render_bool, return_type=bool, when_used="json"),
]

NonNegativeCount = Annotated[
    int,
    PlainValidator(parse_count, json_schema_input_type=int),
    PlainSerializer(render_int, return_type=int, when_used="json"),
]

PositiveCount = Annotated[
    int,
    PlainValidator(parse_count, json_schema_input_type=int),
    AfterValidator(check_positive_count),
    PlainSerializer(render_int, return_type=int, when_used="json"),
]

OptionalPositiveCount = Annotated[
    int | None,
    PlainValidator(parse_optional_count, json_schema_input_type=int | None),
    AfterValidator(check_optional_positive_count),
    PlainSerializer(render_optional_int, return_type=int | None, when_used="json"),
]

# Identifiers ----------------------------------------------------------------
UserId = Annotated[str, StringConstraints(pattern=r"^user_[0-9]+$")]
RequestId = Annotated[str, StringConstraints(pattern=r"^request_[0-9]+$")]
EventId = Annotated[str, StringConstraints(pattern=r"^event_[0-9]+$")]
PaymentOptionId = Annotated[str, StringConstraints(pattern=r"^payment_option_[0-9]+$")]
MessageId = Annotated[str, StringConstraints(pattern=r"^message_[0-9]+$")]
ImageId = Annotated[str, StringConstraints(pattern=r"^image_[0-9]+$")]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
MachineKey = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_.:-]{0,159}$")]
"""Deterministic machine-generated key (series keys, candidate ids, claim ids)."""

OptionalRequestId = Annotated[RequestId | None, BeforeValidator(blank_to_none)]
OptionalEventId = Annotated[EventId | None, BeforeValidator(blank_to_none)]
OptionalPaymentOptionId = Annotated[PaymentOptionId | None, BeforeValidator(blank_to_none)]

# Text -----------------------------------------------------------------------
ShortText = Annotated[str, StringConstraints(min_length=1, max_length=200)]
UntrustedText = Annotated[str, StringConstraints(min_length=1, max_length=4000)]
"""Free text from users, messages, or images. Data only: never interpreted as instructions."""
ExplanationText = Annotated[
    str, StringConstraints(min_length=1, max_length=600, pattern=r"^[^\r\n]+$")
]


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------
def id_ordinal(identifier: str) -> int:
    """Numeric suffix of a dataset id, for ordering (``payment_option_9`` < ``..._10``)."""
    match = _ID_SUFFIX.search(identifier)
    if match is None:
        raise ValueError("identifier has no numeric suffix")
    return int(match.group(1))


def format_amount(value: Decimal) -> str:
    """Normalised amount text: ``603.30`` -> ``'603.3'``, ``873000.00`` -> ``'873000'``."""
    text = format(quantize_money(value), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def format_plan_amount(value: Decimal) -> str:
    """Payment-plan amount text: integral -> ``'25256'``, otherwise two places -> ``'620.40'``."""
    quantized = quantize_money(value)
    if quantized == quantized.to_integral_value():
        return format(quantized.quantize(Decimal(1)), "f")
    return format(quantized, "f")
