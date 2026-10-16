"""Serving-time scorer: one customer-week in, calibrated risk + explanation out."""
import json

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
TIMELINE_FIELDS = [
    "salary_delay_days", "avg_daily_balance_inr", "credit_card_utilization_pct",
    "upi_to_lending_apps_count", "savings_drawdown_pct", "net_cashflow_ratio",
    "emi_bounced_flag", "missed_emi_count_rolling", "financial_stress_queries",
]


class Scorer:
    def __init__(self):
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
        b = self.b
        hist = self.history(customer_id, week_idx, overrides)
        feats = b["codec"].transform(features.add_trend_features(hist))
        row = feats.iloc[[-1]]
        w = int(row[C.WEEK_IDX].iloc[0])

        # Tabular model + exact TreeSHAP
        X = row[b["tab_cols"]]
        p_lgb = float(self.booster.predict(X)[0])
        contrib = self.booster.predict(X, pred_contrib=True)[0]
        raw_row = hist.iloc[-1].to_dict()
        raw_row.update({c: row[c].iloc[0] for c in b["tab_cols"] if c not in raw_row})
        lgb_f, _ = explain.lgb_factors(contrib, b["tab_cols"], raw_row)

        # Causal sequence model over weeks <= w
        Xw, Mw, _, _ = features.build_windows(hist, b["scaler"], b["seq_len"])
        x, m = Xw[-1], Mw[-1]
        with torch.no_grad():
            logit_g, attn = self.seq(torch.from_numpy(x[None]), torch.from_numpy(m[None]), return_attn=True)
        p_gru = float(torch.sigmoid(logit_g)[0])
        attn = attn[0].numpy() if attn is not None else np.zeros(len(m))

        # History-aware stack + calibration
        Z = stacking.meta_features([p_lgb], [p_gru], [w])
        s = float(b["meta"].predict_proba(Z)[0, 1])
        risk = float(b["calibrator"].predict([s])[0])

        coef = dict(zip(stacking.META_FEATURES, b["meta"].coef_[0]))
        early = w < stacking.EARLY_WEEKS
        w_lgb = coef["lgb_logit"] + (coef["lgb_logit x early"] if early else 0.0)
        w_gru = coef["gru_logit"] + (coef["gru_logit x early"] if early else 0.0)

        n_real = int(m.sum())
        week_numbers = list(range(w - n_real + 2, w + 2))  # 1-based week labels of real steps
        seq_f, week_imp = [], []
        if explain_seq:
            attr = explain.integrated_gradients(self.seq, x, m)
            seq_f = explain.seq_factors(attr, b["seq_cols"], hist.iloc[-1])
            per_week = np.abs(attr).sum(1)[-n_real:]
            tot = per_week.sum() or 1.0
            week_imp = [{"week": wk, "ig_share": float(v / tot), "attention": float(a)}
                        for wk, v, a in zip(week_numbers, per_week, attn[-n_real:])]
            week_imp.sort(key=lambda d: -d["ig_share"])

        # Merge both models' factors into final log-odds space using the meta weights.
        combined = {}
        for f, wt, src in [(f, w_lgb, "LightGBM") for f in lgb_f] + [(f, w_gru, "GRU") for f in seq_f]:
            key = f["feature"]
            c = combined.setdefault(key, {**f, "contribution": 0.0, "sources": []})
            c["contribution"] += float(wt * f["contribution"])
            c["sources"].append(src)
        combined = sorted(combined.values(), key=lambda d: -abs(d["contribution"]))
        for c in combined:
            c["direction"] = "increases_risk" if c["contribution"] > 0 else "decreases_risk"

        fired = prefilter.fired_rules(hist.iloc[[-1]]).iloc[0]
        rules = [k for k, v in fired.items() if bool(v)]
        label = hist[C.TARGET].iloc[-1]
        return {
            "customer_id": customer_id,
            "week_idx": w,
            "week": hist[C.WEEK].iloc[-1].strftime("%Y-%m-%d"),
            "risk": round(risk, 4),
            "tier": self.tier(risk),
            "p_lgb": round(p_lgb, 4),
            "p_gru": round(p_gru, 4),
            "stack_raw": round(s, 4),
            "weights": {"lgb": float(w_lgb), "gru": float(w_gru), "early_history": early},
            "thresholds": {"medium": self.thr["medium"], "high": self.thr["high"]},
            "factors": {"combined": combined[:8], "lgb": lgb_f, "gru": seq_f},
            "week_importance": week_imp,
            "prefilter": {"passed": bool(rules), "rules_fired": rules,
                          "cascade_runs_gru": p_lgb >= b["cascade_threshold"]},
            "actual_label": None if overrides else int(label),
            "is_test_customer": customer_id in set(b["test_customers"]),
            "overrides": overrides or {},
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
            "is_test_customer": customer_id in set(self.b["test_customers"]),
        }
