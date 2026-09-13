"""Score the pipeline against the 25 solved samples.

    python code/evaluation/main.py

Runs the same CLI pipeline on ``sample_requests.csv`` (writing ``evaluation/sample_output.csv``)
and reports per-field agreement with the published answers.
"""

import csv
import json
import sys
from decimal import Decimal
from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parent
CODE_ROOT = EVALUATION_DIR.parent
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import main as cli  # noqa: E402

FIELDS = (
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
)


def score(predictions: Path, labels: Path) -> dict[str, object]:
    with predictions.open(encoding="utf-8", newline="") as handle:
        predicted = {row["request_id"]: row for row in csv.DictReader(handle)}
    with labels.open(encoding="utf-8", newline="") as handle:
        expected = list(csv.DictReader(handle))
    totals = dict.fromkeys((*FIELDS, "amount_safe_to_pay_within_1pct"), 0)
    for row in expected:
        guess = predicted[row["request_id"]]
        for name in FIELDS:
            totals[name] += guess[name] == row[name]
        tolerance = Decimal(row["requested_amount"]) * Decimal("0.01")
        gap = abs(Decimal(guess["amount_safe_to_pay"]) - Decimal(row["amount_safe_to_pay"]))
        totals["amount_safe_to_pay_within_1pct"] += gap <= tolerance
    return {"samples": len(expected), "matches": totals}


def main() -> int:
    output = EVALUATION_DIR / "sample_output.csv"
    exit_code = cli.main(
        [
            "--requests",
            "sample_requests.csv",
            "--output",
            str(output),
            "--usage-report",
            str(EVALUATION_DIR / "sample_usage_report.md"),
        ]
    )
    if exit_code != 0:
        return exit_code
    report = score(output, REPO_ROOT / "dataset" / "sample_requests.csv")
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
