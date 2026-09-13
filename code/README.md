# Buy or Wait? — deterministic multi-agent affordability engine

Every numeric decision comes from deterministic Python tools using `decimal.Decimal`. Agents
orchestrate, verify and format results; they never calculate or invent financial facts. The only
optional LLM use is transcribing amounts from images.

## Setup

```bash
cd <repo root>
python -m pip install -r code/requirements.txt
cp .env.example .env        # optional: set BOW_LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY
```

Python 3.11+ (developed on 3.14). No market-data, banking, web-search or exchange-rate APIs are
called; all inputs come from `dataset/`.

## Run

```bash
python code/main.py                     # full dataset -> output.csv (repo root)
python code/evaluation/main.py          # score against the 25 solved samples
```

`main.py` writes `output.csv` and `code/evaluation/usage_report.md`, prints a JSON summary and
emits structured JSON logs with request IDs. The logs never contain money, keys or tokens. Exit
code `0` means every row passed the output validation policy.

## Architecture

```
dataset/*.csv ──► data/loader.py (strict Pydantic parsing of every row)
                     │
                     ▼
LangGraph (agents/graph.py), async, one pure routing function
  planner ─► perception ─► reasoner ─► verifier ─route()─► output
                              ▲           │
                              └─ reflect ◄┤ (max 3 proposals)
                                          └─► fallback (not_recommended)
                     │
                     ▼
output/batch.py ─► eligibility matrix + template check + CSV validation ─► output.csv
```

| Layer | Package | Responsibility |
|---|---|---|
| Contracts | `schemas/` | Strict, frozen Pydantic models; Decimal money (float rejected, `FloatOperation` trapped); tool I/O contracts; `PlannerState` state machine with a bounded reflection budget |
| Engine | `engine/` | FX at the exact settlement-date rate (missing rate → error), banker's rounding, inclusion/exclusion rules, recurrence detection, 90-day daily trajectory, safe amount, earliest full-payment date, plan viability, spending-change search, six-key ranking |
| Agents | `agents/` | Perception (message rules with verbatim quotes, image amounts, worst-case historical fallback, scam rejection), ReAct reasoner, independent verifier, fallback |
| Output | `output/` | Deterministic string templates, eligibility matrix, strict row/file validation, atomic CSV write |
| Quality | `quality/gates.py`, `tests/` | AST gates (float ban, determinism, network isolation, secret scan, schema strictness), 142 tests |

### Key decision rules (calibrated on the solved samples)

- **Payday rule:** on each day, scheduled debits leave before credits arrive, but a recommended payment is made after that day's credits. A payment on payday can use that day's salary.
- **Safe amount and earliest date:** each day has a payment capacity, `capacity[i] = min(closing[i], min(intraday_low[i+1:]))`.
  - `amount_safe_to_pay = clamp(capacity[0] − minimum, 0, requested)`
  - `earliest_date_for_full_payment` is the first day where `capacity[i] − minimum ≥ requested`.
- **What counts as cash:**
  - Excluded: pending credits, failed or cancelled records, duplicate pending charges, and unrealized investment values. Blank amounts are flagged, never treated as zero.
  - Recurring series need at least two monthly occurrences.
  - Groceries, transport and dining are forecast at their observed cadence and mean amount.
  - A confirmed scheduled salary continues monthly.
- **Plan ranking:** plans are ranked by completing on time, needing no spending changes, lowest total, earliest start, fewest payments, then lowest option id. Only viable plans that finish by the deadline are selected. A verifier rejection excludes that candidate; the third rejection produces `not_recommended`.

## Quality gates

```bash
cd code
python -m pytest        # 142 tests
python -m mypy          # strict, pydantic plugin
python -m ruff check .
python quality/gates.py
```

## Security

Secrets are read only from environment variables or `.env` (gitignored) and held as `SecretStr`.
`.env.example` contains placeholders only.
