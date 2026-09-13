"""Render ``evaluation/usage_report.md`` for the run that produced ``output.csv``."""

from decimal import Decimal
from pathlib import Path

from buy_or_wait.llm.usage import UsageTracker
from buy_or_wait.schemas.primitives import decimal_policy


def _money(value: Decimal) -> str:
    return f"${value.quantize(Decimal('0.000001'))}"


def render_usage_report(
    usage: UsageTracker,
    *,
    requests: int,
    provider_setting: str,
    fallback_rows: int,
    proposals: int,
    duration_ms: int,
) -> str:
    models = usage.snapshot()
    total_in = sum(model.input_tokens for model in models)
    total_out = sum(model.output_tokens for model in models)
    total_calls = sum(model.calls for model in models)
    cache_hits = sum(model.cache_hits for model in models)
    with decimal_policy():
        total_cost = sum((model.cost() for model in models), start=Decimal(0))
        per_request = total_cost / Decimal(requests) if requests else Decimal(0)
        avg_tokens = Decimal(total_in + total_out) / Decimal(requests) if requests else Decimal(0)
    lines = [
        "# Token Usage and Cost Report",
        "",
        "Final full-dataset run that produced `output.csv`.",
        "",
        "## Run summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Requests processed | {requests} |",
        f"| LLM provider setting (`BOW_LLM_PROVIDER`) | `{provider_setting}` |",
        f"| Reasoner proposals (deterministic, no tokens) | {proposals} |",
        f"| Fallback rows (not_recommended) | {fallback_rows} |",
        f"| Wall-clock duration | {duration_ms} ms |",
        "",
        "## Per-model usage",
        "",
        "| Provider | Model | Model calls | Cache hits | Input tokens | Output tokens "
        "| Total tokens | Estimated cost |",
        "|---|---|---|---|---|---|---|---|",
    ]
    if models:
        lines.extend(
            f"| {m.provider} | `{m.model}` | {m.calls} | {m.cache_hits} | {m.input_tokens} "
            f"| {m.output_tokens} | {m.input_tokens + m.output_tokens} | {_money(m.cost())} |"
            for m in models
        )
    else:
        lines.append("| none | none | 0 | 0 | 0 | 0 | 0 | $0.000000 |")
    lines += [
        "",
        "## Overall totals",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Model calls | {total_calls} |",
        f"| Cached responses reused | {cache_hits} |",
        f"| Input tokens | {total_in} |",
        f"| Output tokens | {total_out} |",
        f"| Total tokens | {total_in + total_out} |",
        f"| Average tokens per request | {avg_tokens.quantize(Decimal('0.01'))} |",
        f"| Estimated total cost | {_money(total_cost)} |",
        f"| Estimated cost per request | {_money(per_request)} |",
        "",
        "## Notes",
        "",
        "- All arithmetic, forecasting, plan ranking, verification, and explanation text are "
        "deterministic Python tools; they consume no tokens.",
        "- The only LLM use is optional image transcription of blank amounts (Claude vision via "
        "the Anthropic SDK). When `BOW_LLM_PROVIDER=none` or no key is configured, blank amounts "
        "use the worst-case historical amount and no model is called.",
        "- Prices: Anthropic list prices per million tokens (claude-opus-5 $5 input / $25 output).",
        "- No API keys or credentials are included in this report.",
        "",
    ]
    return "\n".join(lines)


def write_usage_report(destination: Path, content: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8", newline="\n")
