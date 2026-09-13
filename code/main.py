"""Buy or Wait? command-line entry point.

    python code/main.py                       # full dataset -> output.csv
    python code/main.py --requests sample_requests.csv --output code/evaluation/sample_output.csv

Ingestion, agent graph execution, deterministic output validation, and CSV writing all happen
here. Exit code 0 means every row passed the output validation policy.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from buy_or_wait.agents.graph import AffordabilityGraph  # noqa: E402
from buy_or_wait.agents.perception import PerceptionAgent, VisionTranscriber  # noqa: E402
from buy_or_wait.agents.reasoner import ReasoningAgent  # noqa: E402
from buy_or_wait.agents.verifier import VerificationAgent  # noqa: E402
from buy_or_wait.config import AppSettings, LlmProvider  # noqa: E402
from buy_or_wait.data.loader import DatasetRepository  # noqa: E402
from buy_or_wait.llm.report import render_usage_report, write_usage_report  # noqa: E402
from buy_or_wait.llm.usage import UsageTracker  # noqa: E402
from buy_or_wait.observability.structured_logging import (  # noqa: E402
    configure_logging,
    get_logger,
    log_event,
)
from buy_or_wait.output.batch import (  # noqa: E402
    DEFAULT_CONCURRENCY,
    BatchRunner,
    reread_and_validate,
    write_output,
)
from buy_or_wait.output.validation import OutputValidationError  # noqa: E402
from buy_or_wait.schemas.primitives import install_decimal_policy  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Buy or Wait? affordability decisions")
    parser.add_argument("--requests", default="requests.csv", help="request file in the dataset")
    parser.add_argument("--output", type=Path, default=None, help="output CSV path")
    parser.add_argument(
        "--usage-report", type=Path, default=CODE_ROOT / "evaluation" / "usage_report.md"
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    return parser.parse_args(argv)


def build_vision(settings: AppSettings, usage: UsageTracker) -> VisionTranscriber | None:
    if settings.llm_provider is not LlmProvider.ANTHROPIC or settings.anthropic_api_key is None:
        return None
    from buy_or_wait.llm.anthropic_gateway import AnthropicVisionGateway

    return AnthropicVisionGateway(
        api_key=settings.anthropic_api_key.get_secret_value(),
        model=settings.evidence_model,
        cache_dir=settings.llm_cache_dir,
        usage=usage,
    )


async def run(args: argparse.Namespace) -> int:
    install_decimal_policy()
    settings = AppSettings.from_environment()
    logger = configure_logging(settings.log_level, log_file=settings.log_file)
    log = get_logger("cli")
    usage = UsageTracker()
    repository = DatasetRepository.load(settings.dataset_dir, args.requests)
    graph = AffordabilityGraph(
        PerceptionAgent(settings.dataset_dir, build_vision(settings, usage)),
        ReasoningAgent(settings.forecast_horizon_days),
        VerificationAgent(settings.forecast_horizon_days),
    )
    runner = BatchRunner(repository, graph, settings.forecast_horizon_days, args.concurrency)
    try:
        result = await runner.run()
    except OutputValidationError as error:
        log_event(log, "output.invalid", fields={"record_count": len(error.violations)})
        sys.stderr.write(f"{error}\n")
        return 2
    destination: Path = args.output or settings.output_path
    write_output(result.rows, destination)
    reread_and_validate(destination, repository.request_order)
    write_usage_report(
        args.usage_report,
        render_usage_report(
            usage,
            requests=len(result.rows),
            provider_setting=settings.llm_provider.value,
            fallback_rows=result.fallback_count,
            proposals=result.proposals,
            duration_ms=result.duration_ms,
        ),
    )
    summary = {
        "rows": len(result.rows),
        "output": str(destination),
        "fallback_rows": dict(sorted(result.fallback_reasons.items())),
        "proposals": result.proposals,
        "duration_ms": result.duration_ms,
    }
    log_event(
        log,
        "run.completed",
        fields={"record_count": len(result.rows), "duration_ms": result.duration_ms},
    )
    logger.handlers[0].flush()
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
