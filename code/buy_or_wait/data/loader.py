"""Dataset ingestion: every CSV row is parsed through the strict contracts exactly once.

Rows are grouped by user and request and sorted by stable keys, so the context handed to the
agents is identical regardless of file row order.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from buy_or_wait.schemas.entities import (
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    ImageRecord,
    Message,
    PaymentOption,
    PurchaseRequest,
)
from buy_or_wait.schemas.primitives import id_ordinal
from buy_or_wait.schemas.tools import UserFinancialContext

REQUEST_COLUMNS: Final[tuple[str, ...]] = tuple(PurchaseRequest.model_fields)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True, slots=True)
class DatasetRepository:
    """Read-only, validated view of ``dataset/`` (no external data sources)."""

    root: Path
    profiles: dict[str, FinancialProfile]
    requests: dict[str, PurchaseRequest]
    request_order: tuple[str, ...]
    events_by_user: dict[str, tuple[FinancialEvent, ...]]
    options_by_request: dict[str, tuple[PaymentOption, ...]]
    messages_by_user: dict[str, tuple[Message, ...]]
    images_by_user: dict[str, tuple[ImageRecord, ...]]
    exchange_rates: tuple[ExchangeRate, ...]
    sample_rows: tuple[dict[str, str], ...] = field(default=())

    @classmethod
    def load(cls, root: Path, requests_file: str = "requests.csv") -> "DatasetRepository":
        request_rows = read_rows(root / requests_file)
        sample_rows = tuple(read_rows(root / "sample_requests.csv"))
        requests = {
            row["request_id"]: PurchaseRequest.model_validate(
                {column: row[column] for column in REQUEST_COLUMNS}
            )
            for row in request_rows
        }
        events: dict[str, list[FinancialEvent]] = defaultdict(list)
        for row in read_rows(root / "financial_events.csv"):
            event = FinancialEvent.model_validate(row)
            events[event.user_id].append(event)
        options: dict[str, list[PaymentOption]] = defaultdict(list)
        for row in read_rows(root / "request_payment_options.csv"):
            option = PaymentOption.model_validate(row)
            options[option.request_id].append(option)
        messages: dict[str, list[Message]] = defaultdict(list)
        for row in read_rows(root / "messages.csv"):
            message = Message.model_validate(row)
            messages[message.user_id].append(message)
        images: dict[str, list[ImageRecord]] = defaultdict(list)
        for row in read_rows(root / "images.csv"):
            image = ImageRecord.model_validate(row)
            images[image.user_id].append(image)
        rates = tuple(
            sorted(
                (ExchangeRate.model_validate(row) for row in read_rows(root / "exchange_rates.csv")),
                key=lambda r: (r.rate_date, r.from_currency.value, r.to_currency.value),
            )
        )
        return cls(
            root=root,
            profiles={
                row["user_id"]: FinancialProfile.model_validate(row)
                for row in read_rows(root / "financial_profiles.csv")
            },
            requests=requests,
            request_order=tuple(row["request_id"] for row in request_rows),
            events_by_user={
                user: tuple(sorted(items, key=lambda e: (e.cash_date, id_ordinal(e.event_id))))
                for user, items in events.items()
            },
            options_by_request={
                request: tuple(sorted(items, key=lambda o: id_ordinal(o.payment_option_id)))
                for request, items in options.items()
            },
            messages_by_user={
                user: tuple(sorted(items, key=lambda m: (m.sent_at, id_ordinal(m.message_id))))
                for user, items in messages.items()
            },
            images_by_user={
                user: tuple(sorted(items, key=lambda i: id_ordinal(i.image_id)))
                for user, items in images.items()
            },
            exchange_rates=rates,
            sample_rows=sample_rows,
        )

    def context_for(self, request_id: str) -> UserFinancialContext:
        request = self.requests[request_id]
        user_id = request.user_id
        return UserFinancialContext(
            request_id=request_id,
            request=request,
            profile=self.profiles[user_id],
            events=self.events_by_user.get(user_id, ()),
            payment_options=self.options_by_request.get(request_id, ()),
            messages=self.messages_by_user.get(user_id, ()),
            images=self.images_by_user.get(user_id, ()),
            exchange_rates=self.exchange_rates,
        )
