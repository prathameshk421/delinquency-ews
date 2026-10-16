"""Rule-based stress pre-filter (RBI / NHB early-warning-signal style rules).

Only customers that trip at least one rule are sent to the ML ensemble. The
point of the audit in `audit()` is to *measure* what this costs: how many real
future defaulters the gate silently drops, versus how much scoring it saves.
"""
import numpy as np
import pandas as pd

RULES = {
    "salary_delayed": ("salary_delay_days", ">=", 5),
    "emi_bounced": ("emi_bounced_flag", ">=", 1),
    "missed_emi": ("missed_emi_count_rolling", ">=", 1),
    "lending_app_upi": ("upi_to_lending_apps_count", ">=", 2),
    "high_cc_utilisation": ("credit_card_utilization_pct", ">=", 60),
    "negative_cashflow": ("net_cashflow_ratio", "<=", 0.1),
    "balance_falling": ("balance_trend_pct", "<=", -10),
    "late_bills": ("num_bills_paid_late_last_4w", ">=", 2),
    "savings_drawdown": ("savings_drawdown_pct", ">=", 25),
    "stress_queries": ("financial_stress_queries", ">=", 1),
    "high_emi_burden": ("emi_to_income_ratio", ">=", 0.40),
    "external_shock": ("external_shock_flag", ">=", 1),
}


def fired_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for name, (col, op, v) in RULES.items():
        out[name] = (df[col] >= v) if op == ">=" else (df[col] <= v)
    return pd.DataFrame(out, index=df.index)


def gate(df: pd.DataFrame) -> np.ndarray:
    return fired_rules(df).any(axis=1).to_numpy()


def audit(df: pd.DataFrame, y) -> dict:
    g = gate(df)
    y = np.asarray(y)
    per_rule = fired_rules(df)
    return {
        "routed_to_ensemble_pct": float(g.mean() * 100),
        "compute_saved_pct": float((1 - g.mean()) * 100),
        "defaulters_kept_pct": float(g[y == 1].mean() * 100),
        "defaulters_dropped": int(((~g) & (y == 1)).sum()),
        "defaulters_total": int((y == 1).sum()),
        "rule_hit_rate_pct": {k: float(per_rule[k].mean() * 100) for k in RULES},
        "rule_precision_pct": {
            k: float(y[per_rule[k].to_numpy()].mean() * 100) if per_rule[k].any() else 0.0 for k in RULES
        },
    }
