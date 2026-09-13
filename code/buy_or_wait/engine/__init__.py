"""Deterministic ledger engine: pure functions over the tool contracts, Decimal-only maths.

Pipeline: ``normalize_ledger`` -> ``detect_recurrence`` -> ``project_cash_flows`` ->
``forecast_balances`` -> ``compute_safe_amount`` / ``find_earliest_full_payment_date`` ->
``build_payment_schedule`` -> ``evaluate_plan`` -> ``search_spending_changes`` -> ``rank_plans``.
Every function sorts its inputs by stable keys first, so shuffled inputs give identical outputs.
"""
