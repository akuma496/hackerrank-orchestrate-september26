"""The whole dataset must satisfy the contracts, and sample answers must pass the output gate."""

import pytest

from buy_or_wait.schemas.audit import AuditedDecision
from buy_or_wait.schemas.decision import OUTPUT_COLUMNS, DecisionRecord
from buy_or_wait.schemas.enums import IssueSeverity
from tests.conftest import DATASET_DIR, Dataset, audit_context_for, read_rows, sample_decision


def test_every_dataset_row_validates(dataset: Dataset) -> None:
    assert len(dataset.profiles) == len(read_rows("financial_profiles.csv")) == 275
    assert len(dataset.events) == len(read_rows("financial_events.csv"))
    assert len(dataset.rates) == len(read_rows("exchange_rates.csv"))
    assert len(dataset.requests) == 275
    assert len(dataset.options) == len(read_rows("request_payment_options.csv"))
    assert len(dataset.messages) == len(read_rows("messages.csv"))
    assert len(dataset.images) == len(read_rows("images.csv"))


def test_every_request_has_a_referentially_intact_context(dataset: Dataset) -> None:
    for request_id in dataset.requests:
        context = dataset.context_for(request_id)
        assert context.request.request_id == request_id


def test_blank_amounts_are_flagged_for_image_evidence(dataset: Dataset) -> None:
    needing_evidence = {
        event.event_id for event in dataset.events if event.requires_evidence_amount
    }
    linked_images = {image.related_event_id for image in dataset.images}
    assert needing_evidence
    assert needing_evidence <= linked_images


def test_output_template_columns_match_contract() -> None:
    with (DATASET_DIR / "output.csv").open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert tuple(header) == OUTPUT_COLUMNS


@pytest.mark.parametrize("index", range(25))
def test_sample_rows_round_trip_and_pass_output_gate(dataset: Dataset, index: int) -> None:
    row = dataset.sample_rows[index]
    decision = DecisionRecord.from_csv_row(row)
    assert decision.to_csv_row() == {column: row[column] for column in OUTPUT_COLUMNS}

    audited = AuditedDecision.certify(decision, audit_context_for(dataset, row["request_id"]))
    assert all(issue.severity is IssueSeverity.WARNING for issue in audited.issues)


def test_sample_decision_json_round_trip(dataset: Dataset) -> None:
    decision = sample_decision(dataset, "request_21")
    assert DecisionRecord.model_validate_json(decision.model_dump_json()) == decision
