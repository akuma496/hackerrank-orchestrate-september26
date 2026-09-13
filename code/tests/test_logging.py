import io
import json
import logging
from decimal import Decimal

import pytest

from buy_or_wait.observability.structured_logging import (
    DisallowedLogFieldError,
    LogLevel,
    configure_logging,
    get_logger,
    log_event,
    request_context,
    scrub_text,
)


@pytest.fixture
def capture() -> io.StringIO:
    stream = io.StringIO()
    configure_logging(LogLevel.DEBUG, stream=stream)
    return stream


def _lines(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


def test_logs_are_json_with_request_correlation(capture: io.StringIO) -> None:
    logger = get_logger("tools.forecast")
    with request_context("request_26", run_id="run_2026_09_13"):
        log_event(logger, "tool.completed", fields={"tool": "forecast_balances", "step": 4})
    log_event(logger, "outside.context")
    first, second = _lines(capture)
    assert first["request_id"] == "request_26"
    assert first["run_id"] == "run_2026_09_13"
    assert first["tool"] == "forecast_balances"
    assert first["step"] == 4
    assert second["request_id"] is None


def test_free_text_is_scrubbed_of_amounts_and_secrets(capture: io.StringIO) -> None:
    fake_key = "sk-ant-" + "a1B2c3D4" * 4
    get_logger("agents").warning(
        "user can pay INR 25,256 or 1543.35 on 2024-03-03 with key %s for request_26", fake_key
    )
    (line,) = _lines(capture)
    event = str(line["event"])
    assert "25,256" not in event
    assert "1543.35" not in event
    assert fake_key not in event
    assert "2024-03-03" in event
    assert "request_26" in event


def test_structured_money_can_never_be_logged(capture: io.StringIO) -> None:
    get_logger("tools").info(
        "tool.completed",
        extra={"amount": "25256", "step": Decimal("3"), "balance": Decimal("58481.10")},
    )
    (line,) = _lines(capture)
    assert "amount" not in line
    assert "balance" not in line
    assert "step" not in line
    assert line["redacted_fields"] == 3


def test_unknown_fields_fail_fast() -> None:
    with pytest.raises(DisallowedLogFieldError):
        log_event(get_logger("tools"), "tool.completed", fields={"requested_amount": "1"})


def test_exceptions_are_logged_without_tracebacks(capture: io.StringIO) -> None:
    logger = get_logger("planner")
    try:
        raise ValueError("balance 58481.10 below minimum")
    except ValueError:
        logger.exception("planner.failed")
    (line,) = _lines(capture)
    assert line["exc_type"] == "ValueError"
    assert "58481.10" not in str(line["exc_message"])
    assert "Traceback" not in json.dumps(line)


def test_scrub_text_keeps_identifiers_and_dates() -> None:
    assert scrub_text("event_1816 on 2026-04-15 step 3") == "event_1816 on 2026-04-15 step 3"
    assert "IDR" not in scrub_text("salary IDR 42750000 confirmed")
    assert "42750000" not in scrub_text("salary 42750000 confirmed")


def teardown_module() -> None:
    logging.getLogger("buy_or_wait").handlers.clear()
