import asyncio
from datetime import date
from decimal import Decimal

import pytest

from buy_or_wait.agents.fallback import FallbackReason, fallback_decision
from buy_or_wait.agents.graph import AffordabilityGraph
from buy_or_wait.agents.perception import PerceptionAgent
from buy_or_wait.agents.reasoner import ReasoningAgent
from buy_or_wait.agents.verifier import VerificationAgent
from buy_or_wait.data.loader import DatasetRepository
from buy_or_wait.output.batch import BatchRunner, installment_schedules, render_csv
from buy_or_wait.output.eligibility import ELIGIBILITY_MATRIX, check_eligibility
from buy_or_wait.output.templates import render_decision_explanation
from buy_or_wait.output.validation import validate_file, validate_row
from buy_or_wait.schemas.decision import OUTPUT_COLUMNS, DecisionRecord, PaymentPlan
from buy_or_wait.schemas.enums import AffordabilityStatus, IssueSeverity, PaymentMethod
from tests.conftest import DATASET_DIR

HORIZON = 90


@pytest.fixture(scope="module")
def samples() -> DatasetRepository:
    return DatasetRepository.load(DATASET_DIR, "sample_requests.csv")


def _graph() -> AffordabilityGraph:
    return AffordabilityGraph(
        PerceptionAgent(DATASET_DIR), ReasoningAgent(HORIZON), VerificationAgent(HORIZON)
    )


def test_published_answers_pass_the_eligibility_matrix_and_templates(
    samples: DatasetRepository,
) -> None:
    for row in samples.sample_rows:
        context = samples.context_for(row["request_id"])
        decision = DecisionRecord.from_csv_row(row)
        issues = check_eligibility(decision, context, installment_schedules(context), HORIZON)
        errors = [issue for issue in issues if issue.severity is IssueSeverity.ERROR]
        assert errors == [], row["request_id"]
        alternate_wait_wording = row["decision_explanation"].startswith("Wait until")
        if (
            not decision.spending_changes_needed
            and decision.affordability_status is not AffordabilityStatus.NOT_AFFORDABLE
            and not alternate_wait_wording
        ):
            _, text = render_decision_explanation(decision, context, HORIZON)
            action_sentence = text.split(". ", 1)[0]
            published_action = row["decision_explanation"].split(". ", 1)[0]
            assert action_sentence == published_action, row["request_id"]


def test_matrix_covers_exactly_the_legal_outcomes() -> None:
    methods = {key[1] for key in ELIGIBILITY_MATRIX}
    assert methods == set(PaymentMethod)
    assert (AffordabilityStatus.AFFORDABLE_NOW, PaymentMethod.WAIT, False) not in ELIGIBILITY_MATRIX


def test_batch_rows_are_valid_and_ordered(samples: DatasetRepository) -> None:
    result = asyncio.run(BatchRunner(samples, _graph(), HORIZON, concurrency=4).run())
    assert [row["request_id"] for row in result.rows] == list(samples.request_order)
    assert render_csv(result.rows).splitlines()[0] == ",".join(OUTPUT_COLUMNS)
    assert validate_file(OUTPUT_COLUMNS, result.rows, samples.request_order) == []


def test_file_rules_catch_duplicates_missing_and_foreign_ids(samples: DatasetRepository) -> None:
    order = samples.request_order
    rows = [{"request_id": request_id} for request_id in order]
    duplicated = [*rows, rows[0]]
    rules = {violation.detail for violation in validate_file(OUTPUT_COLUMNS, duplicated, order)}
    assert "request_id must appear exactly once" in rules
    foreign = [*rows[1:], {"request_id": "request_999"}]
    details = {violation.detail for violation in validate_file(OUTPUT_COLUMNS, foreign, order)}
    assert {"request_id is not in requests.csv", "request is missing from output"} <= details


def _valid_row(samples: DatasetRepository, request_id: str) -> dict[str, str]:
    return next(row for row in samples.sample_rows if row["request_id"] == request_id)


@pytest.mark.parametrize(
    ("request_id", "column", "value", "rule"),
    [
        ("request_01", "amount_safe_to_pay", "25256.01", "R2"),
        ("request_01", "amount_safe_to_pay", "-1", "R2"),
        ("request_01", "payment_plan", "soon", "R3"),
        (
            "request_02",
            "payment_plan",
            "2025-08-09:15952906.67|2025-09-08:15952906.67|2025-10-08:15952906.67",
            "R5",
        ),
        ("request_19", "payment_plan", "2024-09-04:28820|2024-09-15:10000", "R6"),
        ("request_01", "decision_explanation", "Trust me, buy it.", "R8"),
    ],
)
def test_row_rules_reject_bad_values(
    samples: DatasetRepository, request_id: str, column: str, value: str, rule: str
) -> None:
    context = samples.context_for(request_id)
    row = {name: _valid_row(samples, request_id)[name] for name in OUTPUT_COLUMNS}
    row[column] = value
    rules = {
        violation.rule
        for violation in validate_row(row, context, installment_schedules(context), HORIZON)
    }
    assert rule in rules


def test_fallback_rows_are_valid_and_templated(samples: DatasetRepository) -> None:
    context = samples.context_for("request_07")
    decision = fallback_decision(context, FallbackReason.PROCESSING_ERROR)
    row = decision.to_csv_row()
    assert row["recommended_payment_method"] == PaymentMethod.NOT_RECOMMENDED.value
    assert row["payment_plan"] == "none"
    assert row["amount_safe_to_pay"] == "0"
    assert validate_row(row, context, installment_schedules(context), HORIZON) == []
    limited = fallback_decision(
        context, FallbackReason.VERIFICATION_LIMIT, Decimal("5000"), date(2024, 10, 23)
    )
    assert (
        validate_row(limited.to_csv_row(), context, installment_schedules(context), HORIZON) == []
    )


def test_partial_payment_must_be_allowed_by_request(samples: DatasetRepository) -> None:
    context = samples.context_for("request_02")
    decision = DecisionRecord(
        request_id="request_02",
        amount_safe_to_pay=Decimal("17229139.2"),
        affordability_status=AffordabilityStatus.AFFORDABLE_WITH_PLAN,
        recommended_payment_method=PaymentMethod.PARTIAL_PAYMENT,
        payment_plan=PaymentPlan.from_field("2025-08-05:17229139.20|2025-09-15:28788860.80"),
        earliest_date_for_full_payment=date(2025, 9, 15),
        spending_changes_needed=(),
        decision_explanation="placeholder",
    )
    details = {issue.detail for issue in check_eligibility(decision, context, (), HORIZON)}
    assert "partial payment must be allowed by both the user and the request" in details
