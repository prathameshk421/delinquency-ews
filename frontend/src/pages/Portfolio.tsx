import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, Portfolio as P, pct } from "../api";
import { Loading, RiskBar, Stat, TierPill } from "../components/ui";

export default function Portfolio() {
  const nav = useNavigate();
  const [data, setData] = useState<P | null>(null);
  const [week, setWeek] = useState<number | undefined>();
  const [tier, setTier] = useState("");
  const [q, setQ] = useState("");
  const [err, setErr] = useState("");

  useEffect(() => {
    const t = setTimeout(() => {
      api.portfolio({ week_idx: week, tier, q }).then(setData).catch((e) => setErr(String(e.message)));
    }, 150);
    return () => clearTimeout(t);
  }, [week, tier, q]);

  if (err) return <div className="page"><div className="empty err">{err}</div></div>;
  if (!data) return <div className="page"><Loading /></div>;

  return (
    <div className="page">
      <div className="page-head">
        <div className="eyebrow">Held-out portfolio · week {data.week_idx + 1}</div>
        <h1>Weekly watch-list</h1>
        <p className="lede">Every held-out customer scored for the selected week, ranked by calibrated risk. Click a row to see why.</p>
      </div>

      <div className="grid g4">
        <Stat label="Customers scored" value={data.total.toLocaleString()} />
        <Stat label="High risk" value={<span style={{ color: "var(--high)" }}>{data.counts.High}</span>} sub="escalate to relationship manager" />
        <Stat label="Medium risk" value={<span style={{ color: "var(--med)" }}>{data.counts.Medium}</span>} sub="digital nudge / check-in" />
        <Stat label="Low risk" value={<span style={{ color: "var(--low)" }}>{data.counts.Low}</span>} sub="monitor" />
      </div>

      <div className="toolbar section" style={{ marginTop: 28, marginBottom: 14 }}>
        <div className="seg">
          {data.weeks_available.map((w) => (
            <button key={w} className={w === data.week_idx ? "on" : ""} onClick={() => setWeek(w)}>W{w + 1}</button>
          ))}
        </div>
        <div className="seg">
          {["", "High", "Medium", "Low"].map((t) => (
            <button key={t} className={tier === t ? "on" : ""} onClick={() => setTier(t)}>{t || "All"}</button>
          ))}
        </div>
        <input className="input" style={{ maxWidth: 220 }} placeholder="Search customer ID" value={q} onChange={(e) => setQ(e.target.value)} />
      </div>

      <div className="card table-wrap" style={{ padding: 6 }}>
        <table className="table">
          <thead>
            <tr><th>Customer</th><th>Segment</th><th>Product</th><th>Risk</th><th className="num">Score</th><th className="num">Δ vs last wk</th><th className="num">LGB / GRU</th><th>Tier</th><th>Actual</th></tr>
          </thead>
          <tbody>
            {data.rows.map((r) => (
              <tr key={r.customer_id} className="click" onClick={() => nav(`/customer/${r.customer_id}?w=${r.week_idx}`)}>
                <td className="mono">{r.customer_id}</td>
                <td>{r.customer_segment}</td>
                <td className="muted">{r.product_type.replace(/_/g, " ")}</td>
                <td style={{ width: 130 }}><RiskBar value={r.risk} tier={r.tier} /></td>
                <td className="num">{pct(r.risk)}</td>
                <td className={`num ${r.delta > 0.005 ? "down" : r.delta < -0.005 ? "up" : "muted"}`}>
                  {r.delta > 0 ? "+" : ""}{(r.delta * 100).toFixed(1)}
                </td>
                <td className="num muted">{r.p_lgb.toFixed(2)} / {r.p_gru.toFixed(2)}</td>
                <td><TierPill tier={r.tier} /></td>
                <td>{r.label ? <span className="pill High">defaulted</span> : <span className="pill neutral">ok</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {data.rows.length === 0 && <div className="empty">No customers match.</div>}
      </div>
      <p className="small muted" style={{ marginTop: 10 }}>Showing top {data.rows.length} by risk. "Actual" is the ground-truth label (shown because this is a held-out evaluation set).</p>
    </div>
  );
}
