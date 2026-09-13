from pathlib import Path

from buy_or_wait.config import AppSettings, LlmProvider
from quality.gates import (
    check_determinism,
    check_float_ban,
    check_network_isolation,
    check_schema_contracts,
    check_secrets,
    scan_engine,
    scan_secrets_in_text,
)


def test_engine_passes_static_gates() -> None:
    assert scan_engine() == []


def test_repository_passes_secret_scan() -> None:
    assert check_secrets() == []


def test_contracts_are_strict_and_complete() -> None:
    assert check_schema_contracts() == []


def test_float_ban_detects_float_usage() -> None:
    source = "import math\nx = 1.5\ny = float('2')\ndef f(a: float) -> None: ...\nz = round(x)\n"
    details = {violation.detail for violation in check_float_ban(source, "sample.py")}
    assert {"imports math", "float literal", "float used outside isinstance"} <= details
    assert "round(); use quantize_money" in details


def test_float_ban_allows_isinstance_rejection() -> None:
    source = (
        "def guard(v: object) -> None:\n    if isinstance(v, float):\n        raise ValueError\n"
    )
    assert check_float_ban(source, "sample.py") == []


def test_determinism_gate_detects_clocks_and_randomness() -> None:
    source = "import random\nfrom datetime import datetime\nstamp = datetime.now()\n"
    details = {violation.detail for violation in check_determinism(source, "sample.py")}
    assert details == {"imports random", ".now() call"}


def test_network_gate_confines_llm_sdks_to_gateway() -> None:
    source = "import requests\nimport anthropic\n"
    outside = check_network_isolation(source, "tools.py", is_llm_gateway=False)
    inside = check_network_isolation(source, "llm/client.py", is_llm_gateway=True)
    assert {v.detail for v in outside} == {"imports requests", "imports anthropic"}
    assert {v.detail for v in inside} == {"imports requests"}


def test_secret_scan_detects_credentials() -> None:
    leaked = "ANTHROPIC_API_KEY=" + "sk-ant-" + "x" * 32
    assert scan_secrets_in_text(leaked, "notes.md")
    assert scan_secrets_in_text("ANTHROPIC_API_KEY=\n", ".env.example") == []


def test_settings_hide_secrets(tmp_path: Path) -> None:
    fake_key = "sk-ant-" + "k" * 24
    settings = AppSettings.from_environment(
        environ={"BOW_LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": fake_key},
        env_file=tmp_path / "missing.env",
    )
    assert settings.llm_provider is LlmProvider.ANTHROPIC
    assert fake_key not in repr(settings)
    assert fake_key not in settings.model_dump_json()
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == fake_key
    assert settings.forecast_horizon_days == 90
