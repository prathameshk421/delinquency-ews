"""Serving-time scorer: one customer-week in, calibrated risk + explanation out."""
import json
import os
import time

# LightGBM and PyTorch each ship an OpenMP runtime; on macOS running both in one process can
# deadlock. Scoring one record at a time needs no parallelism, so pin both to one thread.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402  (import torch before lightgbm)

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from . import config as C
from . import data, explain, features, prefilter, sequence, stacking

# Fields the UI lets you tweak for "what-if" analysis on the current week.
WHAT_IF_FIELDS = [
    "salary_delay_days", "avg_daily_balance_inr", "net_cashflow_ratio", "savings_drawdown_pct",
    "credit_card_utilization_pct", "upi_to_lending_apps_count", "num_bills_paid_late_last_4w",
    "financial_stress_queries", "emi_bounced_flag", "missed_emi_count_rolling",
]
INGEST_SHOW = [
    "salary_delay_days", "avg_daily_balance_inr", "credit_card_utilization_pct",
    "upi_to_lending_apps_count", "emi_due_this_week", "emi_bounced_flag", "missed_emi_count_rolling",
    "financial_stress_queries",
]
FEATURE_SHOW = [
    "weeks_to_emi_due", "salary_delay_days__d1", "avg_daily_balance_inr__rm3",
    "credit_card_utilization_pct__max_sofar", "emi_bounced__cum",
]
ACTIONS = {
    "High": {"action": "Relationship-manager call", "approval": True,
             "why": "High risk: queued for a relationship-manager call. A human must approve before anyone contacts the customer."},
    "Medium": {"action": "Digital nudge", "approval": True,
               "why": "Medium risk: a supportive check-in message is drafted and waits for human approval."},
    "Low": {"action": "Monitor", "approval": False,
            "why": "Low risk: no contact. The customer is re-scored automatically next week."},
}


def _num(v):
    try:
        f = float(v)
        return int(f) if f.is_integer() else round(f, 3)
    except (TypeError, ValueError):
        return str(v)


TIMELINE_FIELDS = [
    "salary_delay_days", "avg_daily_balance_inr", "credit_card_utilization_pct",
    "upi_to_lending_apps_count", "savings_drawdown_pct", "net_cashflow_ratio",
    "emi_bounced_flag", "missed_emi_count_rolling", "financial_stress_queries",
]


class Scorer:
    def __init__(self):
        torch.set_num_threads(1)
        b = joblib.load(C.ARTIFACT_DIR / "bundle.joblib")
        self.b = b
        self.booster = lgb.Booster(model_file=str(C.ARTIFACT_DIR / "lgb_model.txt"))
        self.seq = sequence.ARCHS[b["arch"]](len(b["seq_cols"]), hidden=b["seq_hidden"], dropout=b["seq_dropout"])
        self.seq.load_state_dict(torch.load(C.ARTIFACT_DIR / "seq_model.pt", map_location="cpu"))
        self.seq.eval()
        self.panel = data.load_panel()
        self.by_customer = {k: g.reset_index(drop=True) for k, g in self.panel.groupby(C.ID)}
        self.portfolio = pd.read_json(C.ARTIFACT_DIR / "portfolio.json")
        self.metrics = json.loads((C.REPORT_DIR / "metrics.json").read_text())
        self.thr = b["thresholds"]
        self.test_set = set(b["test_customers"])
        # Warm up torch / LightGBM so the first live request isn't slow.
        self.score(b["test_customers"][0])

    # ------------------------------------------------------------ helpers
    def tier(self, p):
        return "High" if p >= self.thr["high"] else "Medium" if p >= self.thr["medium"] else "Low"

    def history(self, customer_id, week_idx=None, overrides=None):
        if customer_id not in self.by_customer:
            raise KeyError(customer_id)
        h = self.by_customer[customer_id]
        if week_idx is not None:
            h = h[h[C.WEEK_IDX] <= int(week_idx)]
        h = h.copy()
        if overrides:
            last = h.index[-1]
            for k, v in overrides.items():
                if k in WHAT_IF_FIELDS:
                    h.loc[last, k] = float(v)
            # Keep the derived week-on-week velocity columns consistent with the edit.
            derived = {"avg_daily_balance_inr": "balance_velocity", "salary_delay_days": "salary_delay_delta",
                       "upi_to_lending_apps_count": "upi_lending_delta",
                       "savings_drawdown_pct": "savings_drawdown_velocity"}
            for src, dst in derived.items():
                if src in overrides:
                    prev = h[src].iloc[-2] if len(h) > 1 else h.loc[last, src]
                    h.loc[last, dst] = h.loc[last, src] - prev
            if "avg_daily_balance_inr" in overrides:
                h.loc[last, "available_funds_inr"] = max(0.0, h.loc[last, "available_funds_inr"]
                                                         + float(overrides["avg_daily_balance_inr"])
                                                         - float(self.by_customer[h[C.ID].iloc[0]].loc[last, "avg_daily_balance_inr"]))
        return h

    # --------------------------------------------------------------- core
    def score(self, customer_id, week_idx=None, overrides=None, explain_seq=True):
        """Run one customer-week through the full pipeline.

        Every stage is timed and records the real values it consumed and produced in
        `steps`, which the UI shows as the record's trail. Nothing here is simulated.
        """
        b = self.b
        steps = []
        t0 = time.perf_counter()
        clock = [t0]

        def step(sid, title, what, data, status="ok"):
            now = time.perf_counter()
            steps.append({"id": sid, "title": title, "what": what, "status": status,
                          "ms": round((now - clock[0]) * 1000, 2), "data": data})
            clock[0] = now

        # 1. Ingest -------------------------------------------------------
        hist = self.history(customer_id, week_idx, overrides)
        cur = hist.iloc[-1]
        step("ingest", "Weekly record received",
             "The bank's weekly telemetry for this customer arrives, together with their earlier weeks.",
             {"customer_id": customer_id, "week": cur[C.WEEK].strftime("%Y-%m-%d"),
              "week_number": int(cur[C.WEEK_IDX]) + 1, "weeks_of_history": int(len(hist)),
              "raw_fields": int(len(C.BEHAVIOURAL) + len(C.STATIC_NUM_COLS) + len(C.CAT_COLS)),
              "sample": {f: _num(cur[f]) for f in INGEST_SHOW},
              "what_if_overrides": overrides or {}})

        # 2. Feature engineering -----------------------------------------
        feats = b["codec"].transform(features.add_trend_features(hist))
        row = feats.iloc[[-1]]
        w = int(row[C.WEEK_IDX].iloc[0])
        step("features", "Feature engineering",
             "Past-only trend features are computed (week-on-week change, 3-week average, worst-so-far) "
             "plus the EMI calendar position. Only weeks up to this one are used.",
             {"tabular_features": len(b["tab_cols"]), "sequence_features": len(b["seq_cols"]),
              "examples": {explain.label(f): _num(row[f].iloc[0]) for f in FEATURE_SHOW}})

        # 3. Rule pre-filter ---------------------------------------------
        rules = []
        for name, (col, op, thr) in prefilter.RULES.items():
            v = float(cur[col])
            fired = v >= thr if op == ">=" else v <= thr
            rules.append({"rule": name, "field": explain.label(col), "value": round(v, 3),
                          "op": op, "threshold": thr, "fired": bool(fired)})
        fired_names = [r["rule"] for r in rules if r["fired"]]
        step("prefilter", "Early-warning rules",
             f"{len(prefilter.RULES)} RBI-style early-warning rules are checked. A hard gate would skip customers with no "
             "rule fired; here every customer is still scored and the gate is only reported.",
             {"rules": rules, "fired": len(fired_names), "gate": "route" if fired_names else "would skip"},
             status="flag" if fired_names else "ok")

        # 4. LightGBM + TreeSHAP -----------------------------------------
        X = row[b["tab_cols"]]
        p_lgb = float(self.booster.predict(X, num_threads=1)[0])
        contrib = self.booster.predict(X, pred_contrib=True, num_threads=1)[0]
        raw_row = cur.to_dict()
        raw_row.update({c: row[c].iloc[0] for c in b["tab_cols"] if c not in raw_row})
        lgb_f, bias = explain.lgb_factors(contrib, b["tab_cols"], raw_row)
        step("lgb", "LightGBM (current standing)",
             f"A gradient-boosted tree model ({self.booster.num_trees()} trees) scores this week's snapshot. "
             "TreeSHAP splits its log-odds exactly across the features.",
             {"probability": round(p_lgb, 4), "log_odds": round(float(contrib.sum()), 4),
              "baseline_log_odds": round(bias, 4),
              "shap_sum_check": round(float(contrib[:-1].sum() + bias), 4),
              "top_drivers": [{"label": f["label"], "value": f["value"], "shap": round(f["contribution"], 3)}
                              for f in lgb_f[:5]]})

        # 5. Sequence window ---------------------------------------------
        Xw, Mw, _, _ = features.build_windows(hist, b["scaler"], b["seq_len"])
        x, m = Xw[-1], Mw[-1]
        n_real = int(m.sum())
        week_numbers = list(range(w - n_real + 2, w + 2))
        step("window", "Behaviour sequence assembled",
             f"The customer's {'week 1 is' if w == 0 else f'weeks 1–{w + 1} are'} stacked into a {b['seq_len']}×{len(b['seq_cols'])} matrix "
             "(left-padded with zeros if fewer than 12 weeks exist), scaled with training statistics.",
             {"shape": [int(x.shape[0]), int(x.shape[1])], "real_weeks": week_numbers,
              "padded_weeks": int(b["seq_len"] - n_real)})

        # 6. Attention-GRU ----------------------------------------------
        with torch.no_grad():
            logit_g, attn = self.seq(torch.from_numpy(x[None]), torch.from_numpy(m[None]), return_attn=True)
        p_gru = float(torch.sigmoid(logit_g)[0])
        attn = attn[0].numpy() if attn is not None else np.zeros(len(m))
        step("gru", "Attention-GRU (behaviour over time)",
             f"A small recurrent network ({sequence.n_params(self.seq):,} parameters) reads the weeks in order. "
             "Attention shows how much it looked at each past week.",
             {"probability": round(p_gru, 4), "log_odds": round(float(logit_g[0]), 4),
              "attention": [{"week": wk, "weight": round(float(a), 4)} for wk, a in zip(week_numbers, attn[-n_real:])]})

        # 7. Stacking (shown as real arithmetic) ---------------------------
        Z = stacking.meta_features([p_lgb], [p_gru], [w])
        s = float(b["meta"].predict_proba(Z)[0, 1])
        coef = dict(zip(stacking.META_FEATURES, b["meta"].coef_[0]))
        bias_m = float(b["meta"].intercept_[0])
        early = w < stacking.EARLY_WEEKS
        w_lgb = coef["lgb_logit"] + (coef["lgb_logit x early"] if early else 0.0)
        w_gru = coef["gru_logit"] + (coef["gru_logit x early"] if early else 0.0)
        terms = [{"name": n, "x": round(float(v), 4), "coef": round(float(coef[n]), 4),
                  "product": round(float(v * coef[n]), 4)} for n, v in zip(stacking.META_FEATURES, Z[0])]
        z = bias_m + sum(t["product"] for t in terms)
        step("stack", "Meta-learner fuses both models",
             "A logistic regression trained on out-of-fold predictions combines the two models' log-odds. "
             + ("With under 3 weeks of history it trusts LightGBM more." if early
                else "With 3+ weeks of history both models contribute."),
             {"terms": terms, "bias": round(bias_m, 4), "z": round(z, 4), "p_stacked": round(s, 4),
              "effective_weight_lgb": round(w_lgb, 3), "effective_weight_gru": round(w_gru, 3),
              "early_history": early})

        # 8. Calibration ---------------------------------------------------
        risk = float(b["calibrator"].predict([s])[0])
        step("calibrate", "Probability calibration",
             f"{b['calibrator'].kind.capitalize()} scaling (fitted on a separate calibration group of customers) "
             "makes the score an honest probability: 0.30 means about 30 in 100 such customers default.",
             {"method": b["calibrator"].kind, "before": round(s, 4), "after": round(risk, 4)})

        # 9. Tiering -------------------------------------------------------
        tier = self.tier(risk)
        step("tier", "Risk tier",
             f"Medium starts at {self.thr['medium']:.3f} because missing a defaulter is assumed to cost "
             f"{self.thr['cost_fn']:.0f}× a needless call; High starts where past precision reached 80%.",
             {"risk": round(risk, 4), "tier": tier, "medium_threshold": round(self.thr["medium"], 4),
              "high_threshold": round(self.thr["high"], 4)},
             status="flag" if tier != "Low" else "ok")

        # 10. Explanation --------------------------------------------------
        seq_f, week_imp = [], []
        if explain_seq:
            attr = explain.integrated_gradients(self.seq, x, m)
            seq_f = explain.seq_factors(attr, b["seq_cols"], cur)
            per_week = np.abs(attr).sum(1)[-n_real:]
            tot = per_week.sum() or 1.0
            week_imp = [{"week": wk, "ig_share": float(v / tot), "attention": float(a)}
                        for wk, v, a in zip(week_numbers, per_week, attn[-n_real:])]
            week_imp.sort(key=lambda d: -d["ig_share"])
        combined = {}
        for f, wt, src in [(f, w_lgb, "LightGBM") for f in lgb_f] + [(f, w_gru, "GRU") for f in seq_f]:
            c = combined.setdefault(f["feature"], {**f, "contribution": 0.0, "sources": []})
            c["contribution"] += float(wt * f["contribution"])
            c["sources"].append(src)
        combined = sorted(combined.values(), key=lambda d: -abs(d["contribution"]))
        for c in combined:
            c["direction"] = "increases_risk" if c["contribution"] > 0 else "decreases_risk"
        step("explain", "Explanation",
             "TreeSHAP (LightGBM) and Integrated Gradients (GRU) are merged using the meta-learner weights, "
             "giving the reasons behind this score." if explain_seq
             else "TreeSHAP reasons from LightGBM (GRU Integrated Gradients run when the record is opened).",
             {"top_reasons": [{"label": c["label"], "effect": round(c["contribution"], 3), "sources": c["sources"]}
                              for c in combined[:5]]})

        # 11. Routing ------------------------------------------------------
        action = ACTIONS[tier]
        label = None if overrides else int(cur[C.TARGET])
        step("route", "Decision & routing", action["why"],
             {"action": action["action"], "needs_human_approval": action["approval"],
              "actual_outcome": None if label is None else ("defaulted" if label else "did not default")},
             status="flag" if action["approval"] else "ok")

        return {
            "customer_id": customer_id,
            "week_idx": w,
            "week": cur[C.WEEK].strftime("%Y-%m-%d"),
            "risk": round(risk, 4),
            "tier": tier,
            "p_lgb": round(p_lgb, 4),
            "p_gru": round(p_gru, 4),
            "stack_raw": round(s, 4),
            "weights": {"lgb": float(w_lgb), "gru": float(w_gru), "early_history": early},
            "thresholds": {"medium": self.thr["medium"], "high": self.thr["high"]},
            "factors": {"combined": combined[:8], "lgb": lgb_f, "gru": seq_f},
            "week_importance": week_imp,
            "prefilter": {"passed": bool(fired_names), "rules_fired": fired_names,
                          "cascade_runs_gru": p_lgb >= b["cascade_threshold"]},
            "action": action["action"],
            "actual_label": label,
            "is_test_customer": customer_id in self.test_set,
            "overrides": overrides or {},
            "steps": steps,
            "total_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    # ---------------------------------------------------------- read APIs
    def customer(self, customer_id):
        h = self.by_customer[customer_id]
        first = h.iloc[0]
        port = self.portfolio[self.portfolio[C.ID] == customer_id].sort_values(C.WEEK_IDX)
        timeline = []
        for _, r in h.iterrows():
            row = {"week_idx": int(r[C.WEEK_IDX]), "week": r[C.WEEK].strftime("%Y-%m-%d"),
                   "label": int(r[C.TARGET]), **{f: float(r[f]) for f in TIMELINE_FIELDS}}
            pr = port[port[C.WEEK_IDX] == r[C.WEEK_IDX]]
            if len(pr):
                row.update({"risk": float(pr["risk"].iloc[0]), "p_lgb": float(pr["p_lgb"].iloc[0]),
                            "p_gru": float(pr["p_gru"].iloc[0])})
            timeline.append(row)
        return {
            "customer_id": customer_id,
            "profile": {k: (first[k] if isinstance(first[k], str) else float(first[k]))
                        for k in C.STATIC_NUM_COLS + ["customer_segment", "geography_zone", "product_type"]},
            "timeline": timeline,
            "what_if_fields": {f: float(h[f].iloc[-1]) for f in WHAT_IF_FIELDS},
            "is_test_customer": customer_id in self.test_set,
        }
