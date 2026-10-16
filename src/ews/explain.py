"""Explanations: TreeSHAP for LightGBM, Integrated Gradients + attention for the GRU."""
import numpy as np
import torch

LABELS = {
    "age": "Age", "customer_segment": "Customer segment", "geography_zone": "Geography",
    "product_type": "Loan product", "account_vintage_months": "Account vintage",
    "emi_to_income_ratio": "EMI-to-income ratio", "monthly_income_inr": "Monthly income",
    "emi_amount_inr": "EMI amount", "salary_delay_days": "Salary delay (days)",
    "salary_drop_pct": "Salary drop %", "avg_daily_balance_inr": "Avg daily balance",
    "balance_trend_pct": "Balance trend %", "net_cashflow_ratio": "Net cash-flow ratio",
    "savings_drawdown_pct": "Savings drawdown %", "savings_withdrawal_count": "Savings withdrawals",
    "utility_payment_delay_days": "Utility payment delay", "num_bills_paid_late_last_4w": "Bills paid late (4w)",
    "discretionary_spend_inr": "Discretionary spend", "discretionary_vs_4w_avg_pct": "Discretionary vs 4w avg",
    "gambling_lottery_spend_inr": "Gambling / lottery spend", "gambling_4w_change_pct": "Gambling spend change",
    "upi_to_lending_apps_count": "UPI payments to lending apps", "upi_to_lending_apps_amount_inr": "UPI to lending apps (amount)",
    "atm_vs_4w_avg_pct": "ATM withdrawals vs 4w avg", "auto_debit_failures": "Auto-debit failures",
    "credit_card_utilization_pct": "Credit card utilisation", "credit_inquiries_last_30d": "Credit inquiries (30d)",
    "paying_minimum_only_flag": "Paying minimum due only", "mobile_app_logins": "Mobile app logins",
    "financial_stress_queries": "Financial-stress queries", "customer_service_calls": "Customer service calls",
    "emi_due_this_week": "EMI due this week", "available_funds_inr": "Available funds",
    "emi_paid_flag": "EMI paid", "emi_bounced_flag": "EMI bounced", "missed_emi_count_rolling": "Missed EMIs (rolling)",
    "balance_velocity": "Balance velocity", "salary_delay_delta": "Change in salary delay",
    "discretionary_velocity": "Discretionary spend velocity", "upi_lending_delta": "Change in lending-app UPI",
    "savings_drawdown_velocity": "Savings drawdown velocity", "external_shock_flag": "External shock",
    "shock_type": "Shock type", "week_idx": "Weeks of history",
    "emi_bounced__cum": "EMI bounces so far", "shock__cum": "Shocks so far",
    "weeks_to_emi_due": "Weeks until next EMI due",
}
SUFFIX = {"d1": "week-on-week change", "rm3": "3-week average", "ewm": "smoothed trend",
          "vs_first": "change since first week", "max_sofar": "worst so far"}


def label(feat):
    if "__" in feat:
        base, suf = feat.split("__", 1)
        if suf in SUFFIX:
            return f"{LABELS.get(base, base)} ({SUFFIX[suf]})"
    return LABELS.get(feat, feat)


def lgb_factors(contribs, cols, row, k=6):
    """contribs: 1D TreeSHAP vector incl. bias (log-odds)."""
    vals = contribs[:-1]
    order = np.argsort(-np.abs(vals))[:k]
    out = []
    for i in order:
        v = row[cols[i]]
        out.append({
            "feature": cols[i], "label": label(cols[i]),
            "value": v if isinstance(v, str) else float(v),
            "contribution": float(vals[i]),
            "direction": "increases_risk" if vals[i] > 0 else "decreases_risk",
        })
    return out, float(contribs[-1])


def integrated_gradients(model, x, m, steps=32):
    """IG in log-odds space against a mean-behaviour baseline (zeros after scaling).

    x: [T, F] scaled window, m: [T] mask. Returns attributions [T, F].
    """
    model.eval()
    x_t = torch.from_numpy(x).float()
    m_t = torch.from_numpy(m).float()
    base = torch.zeros_like(x_t)
    alphas = torch.linspace(1.0 / steps, 1.0, steps).view(-1, 1, 1)
    path = (base + alphas * (x_t - base)).requires_grad_(True)
    logits = model(path, m_t.unsqueeze(0).expand(steps, -1))
    logits.sum().backward()
    grads = path.grad.mean(0)
    attr = (grads * (x_t - base)).detach().numpy()
    return attr * m[:, None]


def seq_factors(attr, cols, raw_last_row, k=5):
    per_feat = attr.sum(0)
    order = np.argsort(-np.abs(per_feat))[:k]
    return [{
        "feature": cols[i], "label": label(cols[i]),
        "value": float(raw_last_row[cols[i]]),
        "contribution": float(per_feat[i]),
        "direction": "increases_risk" if per_feat[i] > 0 else "decreases_risk",
    } for i in order]
