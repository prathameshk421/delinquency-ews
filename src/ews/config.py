"""Central configuration: paths, column groups and hyper-parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "pre_delinquency_dataset.csv"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

SEED = 42

ID = "customer_id"
WEEK = "observation_week"
TARGET = "will_default_next_2_4_weeks"
WEEK_IDX = "week_idx"  # 0-based position of the row inside the customer's history

CAT_COLS = ["customer_segment", "geography_zone", "product_type", "shock_type"]
STATIC_NUM_COLS = [
    "age",
    "account_vintage_months",
    "emi_to_income_ratio",
    "monthly_income_inr",
    "emi_amount_inr",
]

# The 8 engineered velocity / EMI-event features (the small-input GRU ablation).
VELOCITY_8 = [
    "balance_velocity",
    "salary_delay_delta",
    "discretionary_velocity",
    "upi_lending_delta",
    "savings_drawdown_velocity",
    "emi_due_this_week",
    "emi_paid_flag",
    "emi_bounced_flag",
]

# Every column that changes week to week (behavioural telemetry).
BEHAVIOURAL = [
    "salary_delay_days", "salary_drop_pct", "avg_daily_balance_inr", "balance_trend_pct",
    "net_cashflow_ratio", "savings_drawdown_pct", "savings_withdrawal_count",
    "utility_payment_delay_days", "num_bills_paid_late_last_4w", "discretionary_spend_inr",
    "discretionary_vs_4w_avg_pct", "gambling_lottery_spend_inr", "gambling_4w_change_pct",
    "upi_to_lending_apps_count", "upi_to_lending_apps_amount_inr", "atm_vs_4w_avg_pct",
    "auto_debit_failures", "credit_card_utilization_pct", "credit_inquiries_last_30d",
    "paying_minimum_only_flag", "mobile_app_logins", "financial_stress_queries",
    "customer_service_calls", "emi_due_this_week", "available_funds_inr", "emi_paid_flag",
    "emi_bounced_flag", "missed_emi_count_rolling", "balance_velocity", "salary_delay_delta",
    "discretionary_velocity", "upi_lending_delta", "savings_drawdown_velocity",
    "external_shock_flag",
]

# Sequence-model inputs: behaviour + the EMI schedule phase.
SEQ_COLS = BEHAVIOURAL + ["weeks_to_emi_due"]

# Heavy-tailed money columns get a signed log transform before scaling.
SIGNED_LOG_COLS = [c for c in BEHAVIOURAL if c.endswith("_inr")] + [
    "balance_velocity",
    "discretionary_velocity",
]

# Columns that get lag / rolling / trend features for the tabular model.
TREND_COLS = [
    "salary_delay_days", "avg_daily_balance_inr", "net_cashflow_ratio",
    "savings_drawdown_pct", "credit_card_utilization_pct", "upi_to_lending_apps_count",
    "discretionary_spend_inr", "num_bills_paid_late_last_4w", "mobile_app_logins",
    "financial_stress_queries", "utility_payment_delay_days", "available_funds_inr",
]

# Split fractions (by customer, so nobody appears in two splits).
TEST_FRAC = 0.20
CALIB_FRAC = 0.15
N_FOLDS = 5

LGB_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
)

SEQ_PARAMS = dict(
    hidden=48,
    dropout=0.2,
    lr=2e-3,
    weight_decay=1e-4,
    batch_size=512,
    max_epochs=30,
    patience=4,
)
SEQ_LEN = 12

# Decision policy: missing a real defaulter costs COST_FN, a needless contact costs COST_FP.
COST_FN = 5.0
COST_FP = 1.0
HIGH_TIER_MIN_PRECISION = 0.80
