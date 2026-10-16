import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, Metrics, RULE_LABELS, f3, pct } from "../api";
import { Loading, Section, Stat, chartTooltip } from "../components/ui";

const STEPS = [
  { t: "Customer-level split", d: "65% train · 15% calibration · 20% test, split by customer so no person's future weeks leak into training." },
  { t: "LightGBM", d: "Current-week standing, EMI-schedule phase and causal trend features (lags, 3-week means, EWMA). Exact TreeSHAP." },
  { t: "Causal attention-GRU", d: "Reads only weeks ≤ t of 34 behavioural signals plus the EMI-schedule phase. Attention + Integrated Gradients show which weeks mattered." },
  { t: "History-aware stacking", d: "Logistic meta-learner on out-of-fold log-odds, with an early-history interaction. No in-sample leakage." },
  { t: "Calibrate & tier", d: "Isotonic vs Platt chosen by grouped-CV Brier. Tiers come from a cost ratio, not hand-picked cut-offs." },
];

export default function Pipeline() {
  const [m, setM] = useState<Metrics | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => { api.metrics().then(setM).catch((e) => setErr(String(e.message))); }, []);
  if (err) return <div className="page"><div className="empty err">{err}</div></div>;
  if (!m) return <div className="page"><Loading /></div>;

  const w = m.meta_weights;
  const pf = m.prefilter;
  const rules = Object.keys(pf.rule_hit_rate_pct).map((k) => ({
    rule: RULE_LABELS[k] ?? k, hit: pf.rule_hit_rate_pct[k], precision: pf.rule_precision_pct[k],
  })).sort((a, b) => b.precision - a.precision);
  const fi = m.feature_importance.slice(0, 10).map((f) => ({ ...f, feature: f.feature.replace(/__/g, " · ").replace(/_/g, " ") }));

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Pipeline</h1>
          <p className="lede">How the model works, the experiments behind each choice, and how we tested them.</p>
        </div>
      </div>

      <div className="flow">
        {STEPS.map((s, i) => (
          <div className="card flow-step" key={s.t}>
            <div className="num">{i + 1}</div>
            <h4>{s.t}</h4>
            <p>{s.d}</p>
          </div>
        ))}
      </div>

      <Section title="Learned meta-learner weights" sub="Coefficients on each model's log-odds. The interaction terms switch trust when history is short.">
        <div className="grid g4">
          <Stat label="LightGBM (weeks 3+)" value={f3(w.lgb_logit)} />
          <Stat label="GRU (weeks 3+)" value={f3(w.gru_logit)} />
          <Stat label="LightGBM (weeks 1–2)" value={f3(w.lgb_logit + w["lgb_logit x early"])} />
          <Stat label="GRU (weeks 1–2)" value={f3(w.gru_logit + w["gru_logit x early"])} />
        </div>
      </Section>

      {m.ablations.length > 0 && (
        <Section title="Ablations" sub="Every variant trained on the same customers and scored on the same held-out test set.">
          <div className="card table-wrap">
            <table className="table">
              <thead><tr><th>Model</th><th>Configuration</th><th className="num">Params</th><th className="num">AUC-ROC</th><th className="num">AUC-PR</th></tr></thead>
              <tbody>
                {m.ablations.map((a) => (
                  <tr key={a.model + a.detail} className={a.detail.includes("(main)") ? "highlight" : ""}>
                    <td>{a.model}</td><td className="muted">{a.detail}</td>
                    <td className="num">{a.params ? a.params.toLocaleString() : "—"}</td>
                    <td className="num">{f3(a.auc_roc)}</td><td className="num">{f3(a.auc_pr)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      <Section title="Rule pre-filter vs model cascade" sub="Does gating customers before the ensemble cost us real defaulters?">
        <div className="grid g2">
          <div className="card">
            <h3>Rule-based gate</h3>
            <div className="grid g2" style={{ gap: 10 }}>
              <div><div className="stat-label">Scoring saved</div><div className="stat-value">{pf.compute_saved_pct.toFixed(1)}%</div></div>
              <div><div className="stat-label">Defaulters silently dropped</div><div className="stat-value down">{pf.defaulters_dropped}</div><div className="stat-sub">{(100 - pf.defaulters_kept_pct).toFixed(2)}% of all defaulters</div></div>
            </div>
            <p className="small muted">Pipeline with gate: AUC-ROC {f3(pf.pipeline_with_gate.auc_roc)}, recall {pct(pf.pipeline_with_gate.recall)}.</p>
          </div>
          <div className="card">
            <h3>Model cascade (LightGBM screens, GRU on the rest)</h3>
            <div className="grid g2" style={{ gap: 10 }}>
              <div><div className="stat-label">GRU calls saved</div><div className="stat-value">{m.cascade.gru_calls_saved_pct.toFixed(1)}%</div></div>
              <div><div className="stat-label">Defaulters screened out</div><div className="stat-value">{m.cascade.defaulters_screened_out}</div><div className="stat-sub">target recall {pct(m.cascade.target_recall)}</div></div>
            </div>
            <p className="small muted">Screened-out customers still get a calibrated LightGBM score, so nobody is scored zero. Pipeline AUC-ROC {f3(m.cascade.pipeline.auc_roc)}.</p>
          </div>
        </div>
        <div className="callout" style={{ marginTop: 14 }}>
          <b>Takeaway.</b> A hard rule gate trades real defaulters for a small compute saving. The GRU has only {m.model_sizes.gru_params.toLocaleString()} parameters,
          so the rules are more useful as human-readable early-warning flags than as a gate.
        </div>
        <div className="card table-wrap" style={{ marginTop: 14 }}>
          <table className="table">
            <thead><tr><th>Rule</th><th className="num">Fires on</th><th className="num">Default rate when fired</th></tr></thead>
            <tbody>
              {rules.map((r) => (
                <tr key={r.rule}><td>{r.rule}</td><td className="num">{r.hit.toFixed(1)}%</td><td className="num">{r.precision.toFixed(1)}%</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="What LightGBM relies on" sub="Share of total split gain, top 10 features.">
        <div className="card">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={fi} layout="vertical" margin={{ top: 4, right: 16, bottom: 4, left: 120 }}>
              <CartesianGrid stroke="var(--grid)" horizontal={false} />
              <XAxis type="number" tickFormatter={(v) => `${Math.round(v * 100)}%`} stroke="var(--border-strong)" />
              <YAxis type="category" dataKey="feature" width={220} stroke="var(--border-strong)" />
              <Tooltip {...chartTooltip} formatter={(v: number) => pct(v)} />
              <Bar dataKey="gain_share" name="Gain share" fill="var(--lgb)" radius={[0, 5, 5, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Section>

      <Section title="Data" sub="Synthetic weekly panel modelled on RBI / NHB early-warning-signal guidance.">
        <div className="grid g4">
          <Stat label="Customer-weeks" value={m.data.rows.toLocaleString()} />
          <Stat label="Customers" value={m.data.customers.toLocaleString()} sub={`${m.data.split.train} train · ${m.data.split.calib} calib · ${m.data.split.test} test`} />
          <Stat label="Weeks per customer" value={m.data.weeks} />
          <Stat label="Positive rate" value={pct(m.data.positive_rate)} sub="will default in next 2–4 weeks" />
        </div>
      </Section>
    </div>
  );
}
