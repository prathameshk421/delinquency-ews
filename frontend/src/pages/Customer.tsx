import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import {
  Bar, CartesianGrid, ComposedChart, Legend, Line, ReferenceArea, ReferenceLine,
  ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis,
} from "recharts";
import {
  api, CustomerDetail, FIELD_LABELS, Narrative, PortfolioRow, RULE_LABELS, Scored, pct, tierColor,
} from "../api";
import { FactorList, Loading, Logo, TierPill, chartTooltip, fmtNum } from "../components/ui";

export default function Customer() {
  const { id } = useParams();
  return id ? <CustomerView id={id.toUpperCase()} /> : <Picker />;
}

function Picker() {
  const nav = useNavigate();
  const [q, setQ] = useState("");
  const [top, setTop] = useState<PortfolioRow[]>([]);
  useEffect(() => { api.portfolio({}).then((p) => setTop(p.rows.slice(0, 6))).catch(() => {}); }, []);
  const go = () => q.trim() && nav(`/customer/${q.trim().toUpperCase()}`);
  return (
    <div className="page" style={{ maxWidth: 760, paddingTop: "14vh" }}>
      <div style={{ textAlign: "center", marginBottom: 26 }}>
        <Logo size={44} />
        <h1 style={{ marginTop: 20 }}>Which customer should we look at?</h1>
        <p className="lede" style={{ margin: "0 auto" }}>Score any customer week by week and see what is driving their risk.</p>
      </div>
      <div className="searchbar" style={{ margin: "0 auto" }}>
        <input autoFocus placeholder="Customer ID, e.g. C00033" value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && go()} />
        <button className="btn primary" onClick={go}>Analyse</button>
      </div>
      {top.length > 0 && (
        <div style={{ marginTop: 26, textAlign: "center" }}>
          <div className="small muted" style={{ marginBottom: 10 }}>Highest-risk held-out customers this week</div>
          {top.map((r) => (
            <button key={r.customer_id} className="chip" style={{ cursor: "pointer" }} onClick={() => nav(`/customer/${r.customer_id}`)}>
              <span className="mono">{r.customer_id}</span> · {pct(r.risk, 0)}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function CustomerView({ id }: { id: string }) {
  const nav = useNavigate();
  const [params, setParams] = useSearchParams();
  const [detail, setDetail] = useState<CustomerDetail | null>(null);
  const [scored, setScored] = useState<Scored | null>(null);
  const [base, setBase] = useState<Scored | null>(null);
  const [err, setErr] = useState("");
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [narr, setNarr] = useState<Narrative | null>(null);
  const [narrBusy, setNarrBusy] = useState(false);

  const week = params.get("w") !== null ? Number(params.get("w")) : undefined;

  useEffect(() => {
    setDetail(null); setErr("");
    api.customer(id).then(setDetail).catch((e) => setErr(String(e.message)));
  }, [id]);

  useEffect(() => {
    if (!detail) return;
    const w = week ?? detail.timeline[detail.timeline.length - 1].week_idx;
    setScored(null); setNarr(null); setEdits({});
    api.score(id, w).then((s) => { setScored(s); setBase(s); }).catch((e) => setErr(String(e.message)));
  }, [detail, week, id]);

  const currentRow = useMemo(() => detail?.timeline.find((t) => t.week_idx === scored?.week_idx), [detail, scored]);

  const overrides = () => {
    const o: Record<string, number> = {};
    Object.entries(edits).forEach(([k, v]) => { if (v !== "" && !isNaN(Number(v))) o[k] = Number(v); });
    return o;
  };
  const rescore = async () => {
    if (!scored) return;
    setBusy(true); setNarr(null);
    try { setScored(await api.score(id, scored.week_idx, overrides())); } catch (e) { setErr(String((e as Error).message)); }
    setBusy(false);
  };
  const reset = () => { setEdits({}); setScored(base); setNarr(null); };
  const explainScore = async () => {
    if (!scored) return;
    setNarrBusy(true);
    try { setNarr(await api.narrative(id, scored.week_idx, scored.overrides)); } catch (e) { setNarr({ summary: String((e as Error).message), reason_codes: [], source: "error" }); }
    setNarrBusy(false);
  };

  if (err) return <div className="page"><div className="empty err">{err} <div style={{ marginTop: 12 }}><button className="btn" onClick={() => nav("/customer")}>Back</button></div></div></div>;
  if (!detail) return <div className="page"><Loading label={`Loading ${id}`} /></div>;

  const p = detail.profile;
  const tl = detail.timeline.map((t) => ({ ...t, wk: t.week_idx + 1, defaulted: t.label ? t.risk ?? 0 : null }));
  const whatIf = Object.keys(overrides()).length > 0;

  return (
    <div className="page">
      <div className="page-head" style={{ display: "block" }}>
        <div className="eyebrow">
          <a onClick={() => nav("/customer")} style={{ cursor: "pointer" }}>Customer analysis</a> / {id}
          {!detail.is_test_customer && <span className="pill neutral" style={{ marginLeft: 8 }}>training customer</span>}
        </div>
        <h1>{id}</h1>
        <div>
          {[p.customer_segment, String(p.product_type).replace(/_/g, " "), String(p.geography_zone).replace(/_/g, " "),
            `${p.age} yrs`, `₹${fmtNum(Number(p.monthly_income_inr))}/mo`, `EMI ₹${fmtNum(Number(p.emi_amount_inr))}`,
            `EMI/income ${(Number(p.emi_to_income_ratio) * 100).toFixed(0)}%`, `${p.account_vintage_months} mo vintage`].map((c) => (
            <span className="chip" key={String(c)}>{c}</span>
          ))}
        </div>
      </div>

      <div className="toolbar" style={{ marginBottom: 18 }}>
        <span className="small muted">Observation week</span>
        <div className="seg">
          {detail.timeline.map((t) => (
            <button key={t.week_idx} className={scored?.week_idx === t.week_idx ? "on" : ""} onClick={() => setParams({ w: String(t.week_idx) })}>
              W{t.week_idx + 1}
            </button>
          ))}
        </div>
      </div>

      {!scored ? <Loading label="Scoring" /> : (
        <>
          <div className="grid g3">
            <div className="card">
              <div className="stat-label">Probability of default, next 2–4 weeks {whatIf && <span className="pill Medium" style={{ marginLeft: 6 }}>what-if</span>}</div>
              <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
                <div className="stat-value" style={{ fontSize: 44, color: tierColor(scored.tier) }}>{pct(scored.risk)}</div>
                <TierPill tier={scored.tier} />
              </div>
              {whatIf && base && <div className="stat-sub">was {pct(base.risk)} → <b className={scored.risk > base.risk ? "down" : "up"}>{scored.risk > base.risk ? "+" : ""}{((scored.risk - base.risk) * 100).toFixed(1)} pts</b></div>}
              <Gauge value={scored.risk} med={scored.thresholds.medium} high={scored.thresholds.high} />
              <div className="small muted" style={{ marginTop: 10 }}>
                Week {scored.week_idx + 1} · {scored.week}
                {scored.actual_label !== null && <> · actual outcome: <b style={{ color: scored.actual_label ? "var(--high)" : "var(--low)" }}>{scored.actual_label ? "defaulted" : "did not default"}</b></>}
              </div>
            </div>

            <div className="card">
              <div className="stat-label">How the two models were fused</div>
              <ModelRow name="LightGBM" sub="current standing + trends" p={scored.p_lgb} w={scored.weights.lgb} color="var(--lgb)" />
              <ModelRow name="Attention-GRU" sub={`reads weeks 1–${scored.week_idx + 1}`} p={scored.p_gru} w={scored.weights.gru} color="var(--gru)" />
              <div className="small muted" style={{ marginTop: 8 }}>
                {scored.weights.early_history ? "Only 1–2 weeks of history" : "Weeks 3+"}: the meta-learner weights{" "}
                {scored.weights.lgb >= scored.weights.gru ? "LightGBM" : "the GRU"} more heavily
                ({Math.max(scored.weights.lgb, scored.weights.gru).toFixed(2)} vs {Math.min(scored.weights.lgb, scored.weights.gru).toFixed(2)} on log-odds).
              </div>
            </div>

            <div className="card">
              <div className="stat-label">Early-warning rules fired ({scored.prefilter.rules_fired.length})</div>
              <div style={{ marginTop: 10 }}>
                {scored.prefilter.rules_fired.length
                  ? scored.prefilter.rules_fired.map((r) => <span className="chip" key={r}>{RULE_LABELS[r] ?? r}</span>)
                  : <span className="small muted">None. The rule gate would skip this customer.</span>}
              </div>
              <div className="small muted" style={{ marginTop: 6 }}>
                Cascade: {scored.prefilter.cascade_runs_gru ? "LightGBM could not rule them out → GRU ran." : "LightGBM screened them out → GRU skipped."}
              </div>
            </div>
          </div>

          <div className="card section" style={{ marginTop: 16 }}>
            <h3>Risk trajectory</h3>
            <ResponsiveContainer width="100%" height={250}>
              <ComposedChart data={tl} margin={{ top: 8, right: 16, bottom: 4, left: -8 }}>
                <CartesianGrid stroke="var(--grid)" vertical={false} />
                <ReferenceArea y1={scored.thresholds.high} y2={1} fill="var(--high-soft)" fillOpacity={0.5} />
                <ReferenceArea y1={scored.thresholds.medium} y2={scored.thresholds.high} fill="var(--med-soft)" fillOpacity={0.5} />
                <XAxis dataKey="wk" tickFormatter={(v) => `W${v}`} stroke="var(--border-strong)" />
                <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} stroke="var(--border-strong)" />
                <Tooltip {...chartTooltip} formatter={(v: number) => (typeof v === "number" ? pct(v) : v)} labelFormatter={(v) => `Week ${v}`} />
                <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: 12 }} />
                <ReferenceLine x={scored.week_idx + 1} stroke="var(--text-3)" strokeDasharray="3 3" />
                <Line dataKey="p_lgb" name="LightGBM" stroke="var(--lgb)" dot={false} strokeWidth={1.4} />
                <Line dataKey="p_gru" name="Attention-GRU" stroke="var(--gru)" dot={false} strokeWidth={1.4} />
                <Line dataKey="risk" name="Final risk" stroke="var(--accent)" strokeWidth={2.6} dot={{ r: 3 }} />
                <Scatter dataKey="defaulted" name="Label = will default" fill="var(--high)" shape="diamond" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <div className="grid g2" style={{ marginTop: 16 }}>
            <div className="card">
              <h3>What is driving this score</h3>
              <p className="small muted" style={{ marginTop: -6 }}>TreeSHAP (LightGBM) and Integrated Gradients (GRU), weighted by the meta-learner into final log-odds. Red raises risk.</p>
              <FactorList factors={scored.factors.combined} />
            </div>
            <div className="card">
              <h3>Which past weeks mattered</h3>
              <p className="small muted" style={{ marginTop: -6 }}>Share of the GRU's Integrated-Gradients attribution per week, with its attention weights.</p>
              <ResponsiveContainer width="100%" height={250}>
                <ComposedChart data={[...scored.week_importance].sort((a, b) => a.week - b.week)} margin={{ top: 8, right: 8, bottom: 4, left: -8 }}>
                  <CartesianGrid stroke="var(--grid)" vertical={false} />
                  <XAxis dataKey="week" tickFormatter={(v) => `W${v}`} stroke="var(--border-strong)" />
                  <YAxis tickFormatter={(v) => `${Math.round(v * 100)}%`} stroke="var(--border-strong)" />
                  <Tooltip {...chartTooltip} formatter={(v: number) => pct(v)} labelFormatter={(v) => `Week ${v}`} />
                  <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="ig_share" name="Attribution share" fill="var(--gru)" radius={[6, 6, 6, 6]} maxBarSize={30} />
                  <Line dataKey="attention" name="Attention" stroke="var(--accent)" dot={{ r: 2.5 }} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="grid g2" style={{ marginTop: 16 }}>
            <div className="card">
              <div className="section-head" style={{ marginBottom: 10 }}>
                <h3 style={{ margin: 0 }}>Analyst note</h3>
                <button className="btn primary" onClick={explainScore} disabled={narrBusy}>{narr ? "Regenerate" : "Explain this score"}</button>
              </div>
              {narrBusy && <div className="typing" style={{ padding: "12px 0" }}><span /><span /><span /></div>}
              {!narrBusy && !narr && <p className="small muted">Turns the drivers above into plain language, reason codes and a suggested next step for the collections team.</p>}
              {!narrBusy && narr && (
                <div className="narrative">
                  <p>{narr.summary}</p>
                  {narr.reason_codes?.length > 0 && <div style={{ marginBottom: 12 }}>{narr.reason_codes.map((r) => <span className="chip" key={r}>{r}</span>)}</div>}
                  {narr.recommended_action && <p><b style={{ fontSize: 13 }}>Suggested action · </b>{narr.recommended_action}</p>}
                  {narr.customer_message && <p className="quote">{narr.customer_message}</p>}
                  <div className="small muted">{narr.source === "template" ? "Template" : `Generated by ${narr.source}`}{narr.note ? ` · ${narr.note}` : ""} · requires human approval before contact.</div>
                </div>
              )}
            </div>

            <div className="card">
              <div className="section-head" style={{ marginBottom: 10 }}>
                <h3 style={{ margin: 0 }}>What-if: edit week {scored.week_idx + 1}</h3>
                <div className="toolbar">
                  <button className="btn" onClick={reset} disabled={!whatIf && scored === base}>Reset</button>
                  <button className="btn primary" onClick={rescore} disabled={busy}>{busy ? "Scoring…" : "Re-score"}</button>
                </div>
              </div>
              <div className="grid g2" style={{ gap: 10 }}>
                {Object.keys(FIELD_LABELS).map((k) => (
                  <div className="field" key={k}>
                    <label>{FIELD_LABELS[k]}</label>
                    <input className="input" inputMode="decimal" value={edits[k] ?? String(currentRow?.[k] ?? detail.what_if_fields[k] ?? "")}
                      onChange={(e) => setEdits({ ...edits, [k]: e.target.value })} />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function Gauge({ value, med, high }: { value: number; med: number; high: number }) {
  return (
    <div style={{ marginTop: 14 }}>
      <div style={{ position: "relative", height: 10, borderRadius: 999, overflow: "hidden", display: "flex" }}>
        <div style={{ width: `${med * 100}%`, background: "var(--low-soft)" }} />
        <div style={{ width: `${(high - med) * 100}%`, background: "var(--med-soft)" }} />
        <div style={{ flex: 1, background: "var(--high-soft)" }} />
      </div>
      <div style={{ position: "relative", height: 14 }}>
        <div style={{ position: "absolute", left: `calc(${value * 100}% - 6px)`, top: -12, width: 12, height: 12, borderRadius: "50%", background: "var(--text)", border: "2px solid var(--surface)" }} />
      </div>
      <div className="small muted" style={{ display: "flex", justifyContent: "space-between" }}>
        <span>Low</span><span>Medium ≥ {pct(med, 0)}</span><span>High ≥ {pct(high, 0)}</span>
      </div>
    </div>
  );
}

function ModelRow({ name, sub, p, w, color }: { name: string; sub: string; p: number; w: number; color: string }) {
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <div><b style={{ fontWeight: 500 }}>{name}</b> <span className="small muted">{sub}</span></div>
        <div className="mono">{pct(p)}</div>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 4 }}>
        <div style={{ flex: 1 }}><div className="riskbar"><span style={{ width: `${p * 100}%`, background: color }} /></div></div>
        <span className="small muted" style={{ width: 92, textAlign: "right" }}>weight {w.toFixed(2)}</span>
      </div>
    </div>
  );
}

