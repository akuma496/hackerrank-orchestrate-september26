# Token Usage and Cost Report

Final full-dataset run that produced `output.csv`.

## Run summary

| Metric | Value |
|---|---|
| Requests processed | 250 |
| LLM provider setting (`BOW_LLM_PROVIDER`) | `anthropic` |
| Reasoner proposals (deterministic, no tokens) | 250 |
| Fallback rows (not_recommended) | 0 |
| Wall-clock duration | 6733 ms |

## Per-model usage

| Provider | Model | Model calls | Cache hits | Input tokens | Output tokens | Total tokens | Estimated cost |
|---|---|---|---|---|---|---|---|
| anthropic | `claude-opus-5` | 11 | 0 | 17059 | 768 | 17827 | $0.104495 |

## Overall totals

| Metric | Value |
|---|---|
| Model calls | 11 |
| Cached responses reused | 0 |
| Input tokens | 17059 |
| Output tokens | 768 |
| Total tokens | 17827 |
| Average tokens per request | 71.31 |
| Estimated total cost | $0.104495 |
| Estimated cost per request | $0.000418 |

## Notes

- All arithmetic, forecasting, plan ranking, verification, and explanation text are deterministic Python tools; they consume no tokens.
- The only LLM use is optional image transcription of blank amounts (Claude vision via the Anthropic SDK). When `BOW_LLM_PROVIDER=none` or no key is configured, blank amounts use the worst-case historical amount and no model is called.
- Prices: Anthropic list prices per million tokens (claude-opus-5 $5 input / $25 output).
- No API keys or credentials are included in this report.
