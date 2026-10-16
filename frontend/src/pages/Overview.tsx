import { useEffect, useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Bar, BarChart,
} from "recharts";
import { api, Metrics, pct, f3 } from "../api";
import { Loading, Section, Stat, chartTooltip } from "../components/ui";

const SERIES = [
  { key: "final", name: "Final (stacked + calibrated)", color: "var(--accent)" },
  { key: "lgb", name: "LightGBM", color: "var(--lgb)" },
  { key: "gru", name: "Attention-GRU", color: "var(--gru)" },
] as const;

export default function Overview() {
  const [m, setM] = useState<Metrics | null>(null);
  const [err, setErr] = useState("");
  const [curve, setCurve] = useState<"pr" | "roc">("pr");

  useEffect(() => { api.metrics().then(setM).catch((e) => setErr(String(e.message))); }, []);
  if (err) return <div className="page"><div className="empty err">{err}</div></div>;
  if (!m) return <div className="page"><Loading /></div>;

  const finalKey = Object.keys(m.results).find((k) => k.startsWith("FINAL"))!;
  const fin = m.results[finalKey];
  const early = m.by_week.filter((b) => b.week <= 2);
  const late = m.by_week.filter((b) => b.week > 2);
  const avg = (xs: typeof m.by_week, k: "lgb" | "gru" | "final") => xs.reduce((a, b) => a + b[k], 0) / Math.max(1, xs.length);

  const curveData = (() => {
    const rows: Record<string, number>[] = [];
    SERIES.forEach((s) => m.curves[s.key][curve].forEach((p) => rows.push({ x: +p.x.toFixed(3), [s.key]: p.y })));
    return rows.sort((a, b) => a.x - b.x);
  })();
  const relData = m.reliability.calibrated.map((r, i) => ({
    pred: +r.pred.toFixed(3), calibrated: r.obs, uncalibrated: m.reliability.uncalibrated[i]?.obs,
  }));
  const lead = Object.entries(m.lead_time.histogram).sort((a, b) => Number(a[0]) - Number(b[0]))
    .map(([k, v]) => ({ weeks: Number(k) >= 8 ? "8+" : k, n: v }));

  return (
    <div className="page">
      <div className="page-head">
        <div className="eyebrow">Loan delinquency early warning · model report</div>
        <h1>Who is likely to miss an EMI in the next 2–4 weeks?</h1>
        <p className="lede">
          LightGBM scores each customer's current financial standing, a causal attention-GRU reads their weekly behaviour
          up to that week, and a history-aware meta-learner fuses both into one calibrated probability. All numbers below are on{" "}
          {m.data.split.test.toLocaleString()} customers the models never saw during training.
        </p>
      </div>

      <div className="grid g4">
        <Stat label="AUC-ROC" value={f3(fin.auc_roc)} sub={<>original deck reported 0.871</>} />
        <Stat label="AUC-PR" value={f3(fin.auc_pr)} sub={<>base rate {pct(m.data.positive_rate)}</>} />
        <Stat label={`Recall @ ${fin.threshold.toFixed(2)}`} value={pct(fin.recall)} sub={<>precision {pct(fin.precision)}</>} />
        <Stat label="Median lead time" value={`${m.lead_time.median_weeks} wks`} sub={<>before first bounced / missed EMI</>} />
      </div>

      <Section title="Model comparison" sub={`Calibration chosen by grouped-CV Brier: ${m.calibration.chosen}. Action threshold = C_FP / (C_FP + C_FN) = ${fin.threshold.toFixed(3)}.`}>
        <div className="card table-wrap" style={{ padding: 6 }}>
          <table className="table">
            <thead>
              <tr><th>Model</th><th className="num">AUC-ROC</th><th className="num">AUC-PR</th><th className="num">Brier ↓</th><th className="num">ECE ↓</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th></tr>
            </thead>
            <tbody>
              {Object.entries(m.results).map(([k, v]) => (
                <tr key={k} className={k === finalKey ? "highlight" : ""}>
                  <td>{k}</td>
                  <td className="num">{f3(v.auc_roc)}</td><td className="num">{f3(v.auc_pr)}</td>
                  <td className="num">{v.brier.toFixed(4)}</td><td className="num">{v.ece.toFixed(4)}</td>
                  <td className="num">{pct(v.precision)}</td><td className="num">{pct(v.recall)}</td><td className="num">{pct(v.f1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <div className="grid g2 section">
        <div className="card">
          <div className="section-head" style={{ marginBottom: 6 }}>
            <h3 style={{ margin: 0 }}>{curve === "pr" ? "Precision–recall" : "ROC"} curve</h3>
            <div className="seg">
              <button className={curve === "pr" ? "on" : ""} onClick={() => setCurve("pr")}>PR</button>
              <button className={curve === "roc" ? "on" : ""} onClick={() => setCurve("roc")}>ROC</button>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={curveData} margin={{ top: 4, right: 12, bottom: 22, left: -8 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="x" type="number" domain={[0, 1]} tickCount={6} stroke="var(--border-strong)"
                label={{ value: curve === "pr" ? "Recall" : "False positive rate", position: "insideBottom", offset: -10, fill: "var(--text-3)", fontSize: 12 }} />
              <YAxis domain={[0, 1]} tickCount={6} stroke="var(--border-strong)" />
              <Tooltip {...chartTooltip} formatter={(v: number) => v.toFixed(3)} />
              <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: 12 }} />
              {SERIES.map((s) => (
                <Line key={s.key} dataKey={s.key} name={s.name} stroke={s.color} dot={false} strokeWidth={s.key === "final" ? 2.4 : 1.6} connectNulls type="monotone" />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3>Reliability — does 0.3 really mean 30%?</h3>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={relData} margin={{ top: 4, right: 12, bottom: 22, left: -8 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="pred" type="number" domain={[0, 1]} tickCount={6} stroke="var(--border-strong)"
                label={{ value: "Predicted probability", position: "insideBottom", offset: -10, fill: "var(--text-3)", fontSize: 12 }} />
              <YAxis domain={[0, 1]} tickCount={6} stroke="var(--border-strong)" />
              <Tooltip {...chartTooltip} formatter={(v: number) => v.toFixed(3)} />
              <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: 12 }} />
              <Line dataKey="pred" name="Perfect calibration" stroke="var(--text-3)" strokeDasharray="4 4" dot={false} />
              <Line dataKey="uncalibrated" name="Stacked (raw)" stroke="var(--lgb)" dot={{ r: 2.5 }} />
              <Line dataKey="calibrated" name={`Stacked + ${m.calibration.chosen}`} stroke="var(--accent)" strokeWidth={2.2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3>AUC by weeks of history — why the meta-learner is history-aware</h3>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={m.by_week} margin={{ top: 4, right: 12, bottom: 22, left: -8 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="week" stroke="var(--border-strong)" label={{ value: "Week of observation", position: "insideBottom", offset: -10, fill: "var(--text-3)", fontSize: 12 }} />
              <YAxis domain={["auto", "auto"]} stroke="var(--border-strong)" tickFormatter={(v) => v.toFixed(2)} />
              <Tooltip {...chartTooltip} formatter={(v: number) => v.toFixed(3)} />
              <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: 12 }} />
              {SERIES.map((s) => <Line key={s.key} dataKey={s.key} name={s.name} stroke={s.color} strokeWidth={s.key === "final" ? 2.4 : 1.6} dot={{ r: 2.5 }} />)}
            </LineChart>
          </ResponsiveContainer>
          <p className="small muted" style={{ margin: "6px 0 0" }}>
            Mean AUC in weeks 1–2: LightGBM {avg(early, "lgb").toFixed(3)} vs GRU {avg(early, "gru").toFixed(3)}; weeks 3–12:
            LightGBM {avg(late, "lgb").toFixed(3)} vs GRU {avg(late, "gru").toFixed(3)}. The GRU has little to read early on, so the
            meta-learner gets an early-history interaction and learns how much to trust each model at each stage.
          </p>
        </div>

        <div className="card">
          <h3>Lead time over the reactive (bounced / missed EMI) baseline</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={lead} margin={{ top: 4, right: 12, bottom: 22, left: -8 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="weeks" stroke="var(--border-strong)" label={{ value: "Weeks earlier than reactive flag", position: "insideBottom", offset: -10, fill: "var(--text-3)", fontSize: 12 }} />
              <YAxis stroke="var(--border-strong)" />
              <Tooltip {...chartTooltip} />
              <Bar dataKey="n" name="Customers" fill="var(--accent)" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <p className="small muted" style={{ margin: "6px 0 0" }}>
            {pct(m.lead_time.flagged_earlier_pct / 100, 0)} of {m.lead_time.customers} customers who later bounced/missed an EMI
            were flagged earlier (mean {m.lead_time.mean_weeks.toFixed(1)} weeks).
          </p>
        </div>
      </div>

      <Section title="Risk tiers" sub={`Medium ≥ ${m.thresholds.medium.toFixed(3)} (cost-optimal, FN costs ${m.thresholds.cost_fn}× a false alarm) · High ≥ ${m.thresholds.high.toFixed(3)} (precision ≥ 80% on calibration split)`}>
        <div className="grid g3">
          {(["High", "Medium", "Low"] as const).map((t) => (
            <div className="card" key={t}>
              <span className={`pill ${t}`}>{t}</span>
              <div className="stat-value" style={{ marginTop: 10 }}>{m.tiers[t].default_rate_pct.toFixed(1)}%</div>
              <div className="stat-sub">actual default rate · {m.tiers[t].share_pct.toFixed(1)}% of customer-weeks</div>
            </div>
          ))}
        </div>
      </Section>

      <p className="small muted section">Report generated {m.generated_at}{m.fast_mode ? " · FAST MODE (smoke run, re-train for final numbers)" : ""}.</p>
    </div>
  );
}
