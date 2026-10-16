export type Tier = "High" | "Medium" | "Low";

export interface Summary {
  auc_roc: number; auc_pr: number; brier: number; ece: number;
  threshold: number; precision: number; recall: number; f1: number;
}
export interface Pt { x: number; y: number }
export interface Metrics {
  generated_at: string;
  fast_mode: boolean;
  data: { rows: number; customers: number; weeks: number; positive_rate: number; split: Record<string, number> };
  results: Record<string, Summary>;
  meta_weights: Record<string, number>;
  calibration: { chosen: string; cv_brier: Record<string, number> };
  thresholds: { medium: number; medium_empirical: number; high: number; cost_fn: number; cost_fp: number };
  tiers: Record<Tier, { share_pct: number; default_rate_pct: number; n: number }>;
  by_week: { week: number; lgb: number; gru: number; final: number }[];
  prefilter: {
    routed_to_ensemble_pct: number; compute_saved_pct: number; defaulters_kept_pct: number;
    defaulters_dropped: number; defaulters_total: number;
    rule_hit_rate_pct: Record<string, number>; rule_precision_pct: Record<string, number>;
    pipeline_with_gate: Summary;
  };
  cascade: { lgb_screen_threshold: number; target_recall: number; gru_calls_saved_pct: number; defaulters_screened_out: number; pipeline: Summary };
  lead_time: { customers: number; median_weeks: number; mean_weeks: number; flagged_earlier_pct: number; histogram: Record<string, number> };
  ablations: { model: string; detail: string; params: number | null; auc_roc: number; auc_pr: number }[];
  feature_importance: { feature: string; gain_share: number }[];
  curves: Record<"final" | "lgb" | "gru", { roc: Pt[]; pr: Pt[] }>;
  reliability: Record<"uncalibrated" | "calibrated", { pred: number; obs: number; n: number }[]>;
  model_sizes: { gru_params: number; lgb_trees: number };
}

export interface PortfolioRow {
  customer_id: string; week_idx: number; week: string; customer_segment: string; geography_zone: string;
  product_type: string; p_lgb: number; p_gru: number; risk: number; tier: Tier; label: number; delta: number;
}
export interface Portfolio { week_idx: number; weeks_available: number[]; counts: Record<Tier, number>; total: number; rows: PortfolioRow[] }

export interface Factor { feature: string; label: string; value: number | string; contribution: number; direction: string; sources?: string[] }
export interface Scored {
  customer_id: string; week_idx: number; week: string; risk: number; tier: Tier;
  p_lgb: number; p_gru: number; stack_raw: number;
  weights: { lgb: number; gru: number; early_history: boolean };
  thresholds: { medium: number; high: number };
  factors: { combined: Factor[]; lgb: Factor[]; gru: Factor[] };
  week_importance: { week: number; ig_share: number; attention: number }[];
  prefilter: { passed: boolean; rules_fired: string[]; cascade_runs_gru: boolean };
  actual_label: number | null; is_test_customer: boolean; overrides: Record<string, number>;
}
export interface TimelineRow { week_idx: number; week: string; label: number; risk?: number; p_lgb?: number; p_gru?: number; [k: string]: number | string | undefined }
export interface CustomerDetail {
  customer_id: string; profile: Record<string, number | string>; timeline: TimelineRow[];
  what_if_fields: Record<string, number>; is_test_customer: boolean;
}
export interface Narrative {
  summary: string; reason_codes: string[]; recommended_action?: string; customer_message?: string; source: string; note?: string;
}
export interface Health { status: string; models_trained: boolean; llm: { configured: boolean; model: string } }

async function req<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, body === undefined ? undefined : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!r.ok) {
    let msg = `${r.status}`;
    try { msg = (await r.json()).detail ?? msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  return r.json();
}

export const api = {
  health: () => req<Health>("/api/health"),
  metrics: () => req<Metrics>("/api/metrics"),
  portfolio: (p: { week_idx?: number; tier?: string; q?: string }) => {
    const s = new URLSearchParams();
    Object.entries(p).forEach(([k, v]) => v !== undefined && v !== "" && s.set(k, String(v)));
    return req<Portfolio>(`/api/portfolio?${s}`);
  },
  customer: (id: string) => req<CustomerDetail>(`/api/customers/${id}`),
  score: (customer_id: string, week_idx?: number, overrides?: Record<string, number>) =>
    req<Scored>("/api/score", { customer_id, week_idx, overrides }),
  narrative: (customer_id: string, week_idx?: number, overrides?: Record<string, number>) =>
    req<Narrative>("/api/narrative", { customer_id, week_idx, overrides }),
};

export const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
export const f3 = (v: number) => v.toFixed(3);
export const tierColor = (t: Tier) => (t === "High" ? "var(--high)" : t === "Medium" ? "var(--med)" : "var(--low)");
export const RULE_LABELS: Record<string, string> = {
  salary_delayed: "Salary delayed ≥ 5 days", emi_bounced: "EMI bounced", missed_emi: "Missed EMI",
  lending_app_upi: "≥ 2 UPI payments to lending apps", high_cc_utilisation: "Card utilisation ≥ 60%",
  negative_cashflow: "Net cash-flow ≤ 10%", balance_falling: "Balance down ≥ 10%", late_bills: "≥ 2 bills paid late",
  savings_drawdown: "Savings drawdown ≥ 25%", stress_queries: "Financial-stress queries", high_emi_burden: "EMI ≥ 40% of income",
  external_shock: "External shock event",
};
export const FIELD_LABELS: Record<string, string> = {
  salary_delay_days: "Salary delay (days)", avg_daily_balance_inr: "Avg daily balance (₹)",
  net_cashflow_ratio: "Net cash-flow ratio", savings_drawdown_pct: "Savings drawdown %",
  credit_card_utilization_pct: "Card utilisation %", upi_to_lending_apps_count: "UPI to lending apps",
  num_bills_paid_late_last_4w: "Bills paid late (4w)", financial_stress_queries: "Stress queries",
  emi_bounced_flag: "EMI bounced (0/1)", missed_emi_count_rolling: "Missed EMIs (rolling)",
};

// ---------------------------------------------------------------- live pipeline
export interface StageSummary { id: string; title: string; ms: number; status: string }
export interface LiveRecord {
  customer_id: string; week_idx: number; week: string; risk: number; tier: Tier; p_lgb: number; p_gru: number;
  rules_fired: number; action: string; actual_label: number | null; total_ms: number; stages: StageSummary[];
  reasons: string[]; case_id: string | null;
}
export interface Step { id: string; title: string; what: string; status: string; ms: number; data: any }
export interface Trace extends Scored { steps: Step[]; total_ms: number; action: string }
export interface Case {
  id: string; customer_id: string; week_idx: number; week: string; risk: number; tier: Tier; action: string;
  status: "pending" | "approved" | "rejected"; reasons: string[]; created: string; updated: string;
  message: Narrative | null; note?: string;
}
export interface AuditEntry { ts: string; type: string; [k: string]: any }

export const live = {
  trace: (customer_id: string, week_idx?: number) => req<Trace>("/api/trace", { customer_id, week_idx }),
  queue: () => req<{ items: Case[]; pending: number }>("/api/queue"),
  decide: (id: string, decision: "approve" | "reject", note = "") => req<Case>(`/api/queue/${id}`, { decision, note }),
  audit: (limit = 40) => req<{ entries: AuditEntry[]; total: number }>(`/api/audit?limit=${limit}`),
};
