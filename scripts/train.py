"""End-to-end training + evaluation.

    python scripts/train.py            # full run (~10-15 min on a laptop CPU)
    python scripts/train.py --fast     # quick smoke run (fewer epochs/folds, no ablations)

Pipeline
  1. Customer-level split: train 65% / calib 15% / test 20%.
  2. LightGBM on current-week + causal trend features  -> out-of-fold (OOF) probs.
  3. Causal attention-GRU on each customer's past weeks -> OOF probs.
  4. Logistic meta-learner on OOF logits (stacking without in-sample leakage).
  5. Isotonic calibration of the stacked score on the untouched calib split.
  6. Cost-based tier thresholds chosen on calib, reported on test.
  7. Ablations (TCN, feature sets, window length), pre-filter audit, cascade, lead time.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import warnings

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ews import config as C  # noqa: E402
from ews import data, features, metrics, prefilter, sequence, stacking, tabular  # noqa: E402

warnings.filterwarnings("ignore")

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def es_split(customers, frac=0.1, seed=C.SEED):
    """Hold aside a slice of *training* customers for early stopping."""
    cs = np.array(sorted(customers))
    rng = np.random.default_rng(seed + 7)
    rng.shuffle(cs)
    k = max(1, int(len(cs) * frac))
    return set(cs[k:]), set(cs[:k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    seq_params = {"max_epochs": 4, "patience": 2} if args.fast else {}
    n_folds = 2 if args.fast else C.N_FOLDS

    C.ARTIFACT_DIR.mkdir(exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    log("loading panel")
    raw = data.load_panel()
    df = features.add_trend_features(raw).sort_values([C.ID, C.WEEK_IDX]).reset_index(drop=True)
    codec = features.CategoryCodec().fit(df)
    df = codec.transform(df)
    y = df[C.TARGET].to_numpy()
    split = data.split_customers(df)
    is_ = {k: df[C.ID].isin(v).to_numpy() for k, v in split.items()}
    log(f"rows={len(df)} customers train/calib/test = "
        f"{len(split['train'])}/{len(split['calib'])}/{len(split['test'])}  pos_rate={y.mean():.3f}")

    tab_cols = features.tabular_columns(with_trends=True)
    tab_cols_raw = features.tabular_columns(with_trends=False)
    seq_cols = C.SEQ_COLS

    def rows(custs):
        return df[C.ID].isin(custs).to_numpy()

    # Window tensors for the main sequence model (scaler fit on train rows only).
    scaler = features.SeqScaler(seq_cols).fit(df[is_["train"]])
    Xw, Mw, yw, _ = features.build_windows(df, scaler, C.SEQ_LEN)

    def fit_seq(train_custs, arch="attn_gru", X=Xw, M=Mw, tag=""):
        tr, es = es_split(train_custs)
        a, b = rows(tr), rows(es)
        log(f"  training {arch}{tag} on {a.sum()} windows")
        model, _ = sequence.train_seq_model(arch, X[a], M[a], yw[a], X[b], M[b], yw[b],
                                            params=seq_params, verbose=False)
        return model

    def fit_tab(train_custs, cols=tab_cols):
        tr, es = es_split(train_custs)
        a, b = rows(tr), rows(es)
        return tabular.fit_lgb(df.loc[a, cols], y[a], df.loc[b, cols], y[b])

    # ------------------------------------------------- OOF for the stacker
    log(f"OOF stacking: {n_folds} customer-grouped folds")
    oof_lgb = np.full(len(df), np.nan)
    oof_gru = np.full(len(df), np.nan)
    for k, fold in enumerate(data.customer_folds(split["train"], n_folds)):
        fit_custs = split["train"] - fold
        f = rows(fold)
        m_lgb = fit_tab(fit_custs)
        oof_lgb[f] = m_lgb.predict_proba(df.loc[f, tab_cols])[:, 1]
        m_gru = fit_seq(fit_custs)
        oof_gru[f] = torch.sigmoid(torch.from_numpy(sequence.predict_logits(m_gru, Xw[f], Mw[f]))).numpy()
        log(f"  fold {k + 1}: LGB AUC={roc_auc_score(y[f], oof_lgb[f]):.4f} "
            f"GRU AUC={roc_auc_score(y[f], oof_gru[f]):.4f}")

    tr = is_["train"]
    wk_all = df[C.WEEK_IDX].to_numpy()
    meta = stacking.fit_meta(oof_lgb[tr], oof_gru[tr], wk_all[tr], y[tr])
    weights = dict(zip(stacking.META_FEATURES, map(float, meta.coef_[0])))
    weights["bias"] = float(meta.intercept_[0])
    log(f"meta-learner weights {weights}")
    # Same meta-learner without the history interaction, for the ablation table.
    meta_plain = LogisticRegression(max_iter=1000).fit(
        np.c_[tabular.logit(oof_lgb[tr]), tabular.logit(oof_gru[tr])], y[tr])

    # --------------------------------------------- final base models (train)
    log("fitting final base models on all training customers")
    lgb_final = fit_tab(split["train"])
    gru_final = fit_seq(split["train"])

    def base_preds(mask):
        p_l = lgb_final.predict_proba(df.loc[mask, tab_cols])[:, 1]
        p_g = torch.sigmoid(torch.from_numpy(sequence.predict_logits(gru_final, Xw[mask], Mw[mask]))).numpy()
        return p_l, p_g

    def stack_raw(p_l, p_g, w):
        return meta.predict_proba(stacking.meta_features(p_l, p_g, w))[:, 1]

    cal, te = is_["calib"], is_["test"]
    pl_cal, pg_cal = base_preds(cal)
    pl_te, pg_te = base_preds(te)
    s_cal, s_te = stack_raw(pl_cal, pg_cal, wk_all[cal]), stack_raw(pl_te, pg_te, wk_all[te])
    s_te_plain = meta_plain.predict_proba(np.c_[tabular.logit(pl_te), tabular.logit(pg_te)])[:, 1]

    # --------------------------------------------------------- calibration
    calibrator, cal_cv = stacking.choose_calibrator(s_cal, y[cal], df.loc[cal, C.ID].to_numpy())
    log(f"calibrator CV Brier {cal_cv} -> using {calibrator.kind}")
    iso_only = stacking.Calibrator("isotonic").fit(s_cal, y[cal])
    platt_only = stacking.Calibrator("platt").fit(s_cal, y[cal])
    iso_lgb = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pl_cal, y[cal])
    p_cal_final = calibrator.predict(s_cal)
    p_te_final = calibrator.predict(s_te)

    # ---------------------------------------------------------- thresholds
    # With calibrated probabilities the Bayes-optimal action threshold is analytic:
    # act when p * C_FN > (1 - p) * C_FP  <=>  p > C_FP / (C_FP + C_FN).
    thr_action = C.COST_FP / (C.COST_FP + C.COST_FN)
    grid = np.unique(np.round(p_cal_final, 4))
    emp_costs = [metrics.expected_cost(y[cal], p_cal_final, t, C.COST_FN, C.COST_FP) for t in grid]
    thr_action_emp = float(grid[int(np.argmin(emp_costs))])
    thr_high = None
    for t in np.sort(grid):
        if t > thr_action and (p_cal_final >= t).sum() >= 50 and \
                precision_score(y[cal], p_cal_final >= t, zero_division=0) >= C.HIGH_TIER_MIN_PRECISION:
            thr_high = float(t)
            break
    thr_high = thr_high if thr_high is not None else 0.75
    thresholds = {"medium": float(thr_action), "medium_empirical": thr_action_emp, "high": thr_high,
                  "cost_fn": C.COST_FN, "cost_fp": C.COST_FP}
    log(f"thresholds: {thresholds}")

    # ---------------------------------------------------- headline results
    yt = y[te]
    naive = 0.6 * pl_te + 0.4 * pg_te  # the original repo's fixed-weight heuristic
    results = {
        "LightGBM (raw features)": None,
        "LightGBM (+ trend features)": metrics.summary(yt, pl_te, thr_action),
        "Attention-GRU (causal)": metrics.summary(yt, pg_te, thr_action),
        "Fixed 0.6/0.4 average (original)": metrics.summary(yt, naive, thr_action),
        "Stacked, plain meta-learner": metrics.summary(yt, s_te_plain, thr_action),
        "Stacked, history-aware (uncalibrated)": metrics.summary(yt, s_te, thr_action),
        "Stacked + isotonic": metrics.summary(yt, iso_only.predict(s_te), thr_action),
        "Stacked + Platt": metrics.summary(yt, platt_only.predict(s_te), thr_action),
    }
    results[f"FINAL: stacked + {calibrator.kind}"] = metrics.summary(yt, p_te_final, thr_action)
    lgb_rawm = fit_tab(split["train"], tab_cols_raw)
    results["LightGBM (raw features)"] = metrics.summary(
        yt, lgb_rawm.predict_proba(df.loc[te, tab_cols_raw])[:, 1], thr_action)
    tab_cols_nosched = features.tabular_columns(with_trends=True, with_schedule=False)
    lgb_ns = fit_tab(split["train"], tab_cols_nosched)
    p_lgb_ns = lgb_ns.predict_proba(df.loc[te, tab_cols_nosched])[:, 1]
    for k, v in results.items():
        log(f"  {k:36s} AUC={v['auc_roc']:.4f} AP={v['auc_pr']:.4f} Brier={v['brier']:.4f} "
            f"ECE={v['ece']:.4f} R@thr={v['recall']:.3f} P@thr={v['precision']:.3f}")

    final_te = p_te_final
    tiers_te = np.where(final_te >= thr_high, "High", np.where(final_te >= thr_action, "Medium", "Low"))
    tier_stats = {}
    for t in ("High", "Medium", "Low"):
        m = tiers_te == t
        tier_stats[t] = {"share_pct": float(m.mean() * 100),
                         "default_rate_pct": float(yt[m].mean() * 100) if m.any() else 0.0,
                         "n": int(m.sum())}

    # ---------------------------------------------------------- per week
    wk = df.loc[te, C.WEEK_IDX].to_numpy()
    by_week = []
    for w in range(C.SEQ_LEN):
        m = wk == w
        if len(np.unique(yt[m])) == 2:
            by_week.append({"week": w + 1,
                            "lgb": float(roc_auc_score(yt[m], pl_te[m])),
                            "gru": float(roc_auc_score(yt[m], pg_te[m])),
                            "final": float(roc_auc_score(yt[m], final_te[m]))})

    # ---------------------------------------------- pre-filter vs cascade
    te_df = df[te]
    audit = prefilter.audit(te_df, yt)
    g = prefilter.gate(te_df)
    gated_score = np.where(g, final_te, 0.0)
    audit["pipeline_with_gate"] = metrics.summary(yt, gated_score, thr_action)

    # Cascade: LightGBM (cheap) scores everyone; GRU only runs if LightGBM can't rule them out.
    target_recall = 0.995
    pl_sorted = np.sort(pl_cal[y[cal] == 1])
    thr_cascade = float(pl_sorted[int((1 - target_recall) * len(pl_sorted))])
    run_gru = pl_te >= thr_cascade
    cascade_score = np.where(run_gru, final_te, iso_lgb.predict(pl_te))
    cascade = {
        "lgb_screen_threshold": thr_cascade,
        "target_recall": target_recall,
        "gru_calls_saved_pct": float((1 - run_gru.mean()) * 100),
        "defaulters_screened_out": int(((~run_gru) & (yt == 1)).sum()),
        "pipeline": metrics.summary(yt, cascade_score, thr_action),
    }
    log(f"rule gate keeps {audit['defaulters_kept_pct']:.2f}% defaulters, saves {audit['compute_saved_pct']:.1f}%; "
        f"cascade saves {cascade['gru_calls_saved_pct']:.1f}% GRU calls, drops {cascade['defaulters_screened_out']}")

    # --------------------------------------------------------- lead time
    # Reactive baseline = first week the bank would see a bounced/missed EMI.
    tmp = te_df[[C.ID, C.WEEK_IDX, "emi_bounced_flag", "missed_emi_count_rolling"]].copy()
    tmp["model_flag"] = final_te >= thr_action
    tmp["reactive_flag"] = (tmp["emi_bounced_flag"] >= 1) | (tmp["missed_emi_count_rolling"] >= 1)
    leads = []
    for _, gdf in tmp.groupby(C.ID):
        r = gdf.loc[gdf["reactive_flag"], C.WEEK_IDX]
        mflag = gdf.loc[gdf["model_flag"], C.WEEK_IDX]
        if len(r) and len(mflag) and r.iloc[0] > 0:
            leads.append(int(r.iloc[0] - mflag.iloc[0]))
    leads = np.array(leads) if leads else np.array([0])
    lead_time = {"customers": int(len(leads)), "median_weeks": float(np.median(leads)),
                 "mean_weeks": float(leads.mean()), "flagged_earlier_pct": float((leads > 0).mean() * 100),
                 "histogram": {str(k): int(v) for k, v in zip(*np.unique(np.clip(leads, -3, 8), return_counts=True))}}

    # ------------------------------------------------------------ ablations
    ablations = []
    if not args.fast:
        log("ablations")

        def eval_seq(model, X, M, name, detail):
            p = torch.sigmoid(torch.from_numpy(sequence.predict_logits(model, X[te], M[te]))).numpy()
            ablations.append({"model": name, "detail": detail, "params": sequence.n_params(model),
                              "auc_roc": float(roc_auc_score(yt, p)),
                              "auc_pr": float(average_precision_score(yt, p))})

        eval_seq(gru_final, Xw, Mw, "Attention-GRU", "34 behavioural + EMI phase, 12-week window (main)")
        tcn = fit_seq(split["train"], "tcn")
        eval_seq(tcn, Xw, Mw, "TCN", "34 behavioural + EMI phase, 12-week window")
        scb = features.SeqScaler(C.BEHAVIOURAL).fit(df[is_["train"]])
        Xb, Mb, _, _ = features.build_windows(df, scb, C.SEQ_LEN)
        mb = fit_seq(split["train"], X=Xb, M=Mb, tag=" (no EMI phase)")
        eval_seq(mb, Xb, Mb, "Attention-GRU", "34 behavioural, no EMI phase, 12-week window")
        del Xb, Mb
        for L in (4, 8):
            Xs, Ms = Xw[:, -L:].copy(), Mw[:, -L:].copy()
            m = fit_seq(split["train"], X=Xs, M=Ms, tag=f" (window {L})")
            eval_seq(m, Xs, Ms, "Attention-GRU", f"34 behavioural + EMI phase, {L}-week window")
            del Xs, Ms
        sc8 = features.SeqScaler(C.VELOCITY_8).fit(df[is_["train"]])
        X8, M8, _, _ = features.build_windows(df, sc8, C.SEQ_LEN)
        m8 = fit_seq(split["train"], X=X8, M=M8, tag=" (velocity-8)")
        eval_seq(m8, X8, M8, "Attention-GRU", "original 8 velocity features, 12-week window")
        del X8, M8
        ablations.append({"model": "LightGBM", "detail": "trend features, no EMI phase", "params": None,
                          "auc_roc": float(roc_auc_score(yt, p_lgb_ns)), "auc_pr": float(average_precision_score(yt, p_lgb_ns))})
        ablations.append({"model": "LightGBM", "detail": "raw current-week features",
                          "params": None, **{k: results["LightGBM (raw features)"][k] for k in ("auc_roc", "auc_pr")}})
        ablations.append({"model": "LightGBM", "detail": "+ causal trend features + EMI phase (main)",
                          "params": None, **{k: results["LightGBM (+ trend features)"][k] for k in ("auc_roc", "auc_pr")}})
        for a in ablations:
            log(f"  {a['model']:14s} {a['detail']:50s} AUC={a['auc_roc']:.4f} AP={a['auc_pr']:.4f}")

    # ------------------------------------------------- global importance
    imp = pd.Series(lgb_final.booster_.feature_importance("gain"), index=tab_cols)
    imp = (imp / imp.sum()).sort_values(ascending=False).head(15)

    # -------------------------------------------------------- save models
    log("saving artifacts")
    lgb_final.booster_.save_model(str(C.ARTIFACT_DIR / "lgb_model.txt"))
    torch.save(gru_final.state_dict(), C.ARTIFACT_DIR / "seq_model.pt")
    joblib.dump({
        "tab_cols": tab_cols, "seq_cols": seq_cols, "seq_len": C.SEQ_LEN, "arch": "attn_gru",
        "seq_hidden": C.SEQ_PARAMS["hidden"], "seq_dropout": C.SEQ_PARAMS["dropout"],
        "codec": codec, "scaler": scaler, "meta": meta, "calibrator": calibrator, "iso_lgb": iso_lgb,
        "thresholds": thresholds, "cascade_threshold": thr_cascade,
        "test_customers": sorted(split["test"]),
    }, C.ARTIFACT_DIR / "bundle.joblib")

    # Portfolio of scored test rows for the dashboard.
    port = te_df[[C.ID, C.WEEK_IDX, "customer_segment", "geography_zone", "product_type"]].copy()
    port["week"] = te_df[C.WEEK].dt.strftime("%Y-%m-%d").to_numpy()
    for c in ("customer_segment", "geography_zone", "product_type"):
        port[c] = port[c].astype(str)
    port["p_lgb"], port["p_gru"], port["risk"] = pl_te, pg_te, final_te
    port["tier"], port["label"] = tiers_te, yt
    port.round(4).to_json(C.ARTIFACT_DIR / "portfolio.json", orient="records")

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "fast_mode": args.fast,
        "data": {"rows": int(len(df)), "customers": int(df[C.ID].nunique()), "weeks": C.SEQ_LEN,
                 "positive_rate": float(y.mean()),
                 "split": {k: len(v) for k, v in split.items()}},
        "results": results,
        "meta_weights": weights,
        "calibration": {"chosen": calibrator.kind, "cv_brier": cal_cv},
        "thresholds": thresholds,
        "tiers": tier_stats,
        "by_week": by_week,
        "prefilter": audit,
        "cascade": cascade,
        "lead_time": lead_time,
        "ablations": ablations,
        "feature_importance": [{"feature": k, "gain_share": float(v)} for k, v in imp.items()],
        "curves": {"final": metrics.curves(yt, final_te), "lgb": metrics.curves(yt, pl_te),
                   "gru": metrics.curves(yt, pg_te)},
        "reliability": {"uncalibrated": metrics.reliability(yt, s_te),
                        "calibrated": metrics.reliability(yt, final_te)},
        "model_sizes": {"gru_params": sequence.n_params(gru_final),
                        "lgb_trees": int(lgb_final.booster_.num_trees())},
    }
    (C.REPORT_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    make_figures(report)
    log("done")


def make_figures(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    c = {"final": "#c15f3c", "lgb": "#5b6b8c", "gru": "#7a9a6a"}

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k, lab in (("final", "Final (stacked + calibrated)"), ("lgb", "LightGBM"), ("gru", "Attention-GRU")):
        ax[0].plot([p["x"] for p in r["curves"][k]["roc"]], [p["y"] for p in r["curves"][k]["roc"]], label=lab, color=c[k])
        ax[1].plot([p["x"] for p in r["curves"][k]["pr"]], [p["y"] for p in r["curves"][k]["pr"]], label=lab, color=c[k])
    ax[0].plot([0, 1], [0, 1], ls="--", color="#bbb")
    ax[0].set(title="ROC (test customers)", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].set(title="Precision-Recall (test customers)", xlabel="Recall", ylabel="Precision")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "roc_pr.png", dpi=160)

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for k, lab, col in (("uncalibrated", "Stacked (raw)", "#999"), ("calibrated", f"Stacked + {r['calibration']['chosen']}", c["final"])):
        pts = r["reliability"][k]
        ax.plot([p["pred"] for p in pts], [p["obs"] for p in pts], marker="o", label=lab, color=col)
    ax.plot([0, 1], [0, 1], ls="--", color="#bbb")
    ax.set(title="Reliability diagram", xlabel="Predicted probability", ylabel="Observed default rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "reliability.png", dpi=160)

    if r["by_week"]:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        w = [b["week"] for b in r["by_week"]]
        for k, lab in (("final", "Stacked"), ("lgb", "LightGBM"), ("gru", "Attention-GRU")):
            ax.plot(w, [b[k] for b in r["by_week"]], marker="o", label=lab, color=c[k])
        ax.set(title="AUC by weeks of history available", xlabel="Week of observation", ylabel="AUC-ROC")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(C.FIG_DIR / "auc_by_week.png", dpi=160)

    fi = r["feature_importance"][::-1]
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.barh([f["feature"] for f in fi], [f["gain_share"] for f in fi], color=c["lgb"])
    ax.set(title="LightGBM global importance (gain share)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "feature_importance.png", dpi=160)
    plt.close("all")


if __name__ == "__main__":
    main()
