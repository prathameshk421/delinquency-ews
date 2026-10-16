import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Area, AreaChart, Cell, Pie, PieChart, ReferenceLine, ResponsiveContainer, YAxis } from "recharts";
import { AuditEntry, Case, LiveRecord, RULE_LABELS, Step, Trace, live, pct, tierColor } from "../api";
import { Loading, TierPill, fmtNum } from "../components/ui";

const STAGES = [
  { id: "ingest", short: "Ingest" },
  { id: "features", short: "Features" },
  { id: "prefilter", short: "Rules" },
  { id: "lgb", short: "LightGBM" },
  { id: "window", short: "Sequence" },
  { id: "gru", short: "GRU" },
  { id: "stack", short: "Stack" },
  { id: "calibrate", short: "Calibrate" },
  { id: "tier", short: "Tier" },
  { id: "explain", short: "Explain" },
  { id: "route", short: "Route" },
];
const SPEEDS = { Slow: 1500, Normal: 700, Fast: 150 } as const;

interface Stats { n: number; ms: number; High: number; Medium: number; Low: number; skip: number; correctHigh: number; stageMs: Record<string, number> }
const emptyStats = (): Stats => ({ n: 0, ms: 0, High: 0, Medium: 0, Low: 0, skip: 0, correctHigh: 0, stageMs: {} });

export default function Live() {
  const nav = useNavigate();
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState<keyof typeof SPEEDS>("Normal");
  const [week, setWeek] = useState(0);
  const [feed, setFeed] = useState<LiveRecord[]>([]);
  const [stats, setStats] = useState<Stats>(emptyStats);
  const [selected, setSelected] = useState<LiveRecord | null>(null);
  const [follow, setFollow] = useState(true);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [traceBusy, setTraceBusy] = useState(false);
  const [pulse, setPulse] = useState(0);
  const [err, setErr] = useState("");
  const es = useRef<EventSource | null>(null);
  const weekRef = useRef(0);
  const busyRef = useRef(false);
  const followRef = useRef(true);
  const reqId = useRef(0);
  useEffect(() => { followRef.current = follow; }, [follow]);

  // Fetch the full trail (with Integrated Gradients) for one record. Stale responses are dropped.
  const loadTrace = useCallback((r: LiveRecord) => {
    const id = ++reqId.current;
    busyRef.current = true;
    setTraceBusy(true);
    live.trace(r.customer_id, r.week_idx)
      .then((t) => { if (id === reqId.current) setTrace(t); })
      .catch((e) => setErr(String(e.message)))
      .finally(() => { if (id === reqId.current) { busyRef.current = false; setTraceBusy(false); } });
  }, []);

  const stop = useCallback(() => { es.current?.close(); es.current = null; setRunning(false); }, []);

  const start = useCallback(() => {
    es.current?.close();
    setErr("");
    const src = new EventSource(`/api/stream?week_idx=${weekRef.current}&delay_ms=${SPEEDS[speed]}`);
    src.addEventListener("week", (e) => { const w = JSON.parse((e as MessageEvent).data).week_idx; weekRef.current = w; setWeek(w); });
    src.addEventListener("record", (e) => {
      const r: LiveRecord = JSON.parse((e as MessageEvent).data);
      setFeed((f) => [r, ...f].slice(0, 60));
      setPulse((p) => p + 1);
      setStats((s) => {
        const stageMs = { ...s.stageMs };
        r.stages.forEach((st) => { stageMs[st.id] = (stageMs[st.id] ?? 0) + st.ms; });
        return {
          n: s.n + 1, ms: s.ms + r.total_ms, High: s.High + (r.tier === "High" ? 1 : 0),
          Medium: s.Medium + (r.tier === "Medium" ? 1 : 0), Low: s.Low + (r.tier === "Low" ? 1 : 0),
          skip: s.skip + (r.rules_fired === 0 ? 1 : 0),
          correctHigh: s.correctHigh + (r.tier === "High" && r.actual_label === 1 ? 1 : 0), stageMs,
        };
      });
      if (followRef.current) {
        setSelected(r);
        if (!busyRef.current) loadTrace(r);
      }
    });
    src.addEventListener("done", () => { weekRef.current = 0; stop(); });
    src.onerror = () => { setErr("Stream interrupted. Is the API running?"); stop(); };
    es.current = src;
    setRunning(true);
  }, [speed, stop, loadTrace]);

  useEffect(() => () => es.current?.close(), []);
  useEffect(() => { if (running) start(); /* restart with new speed */ }, [speed]); // eslint-disable-line react-hooks/exhaustive-deps

  const pick = (r: LiveRecord) => { setFollow(false); setSelected(r); loadTrace(r); };
  const avg = (k: string) => (stats.n ? (stats.stageMs[k] ?? 0) / stats.n : 0);

  const tiers = (["High", "Medium", "Low"] as const).map((t) => ({ t, n: stats[t] }));
  const spark = [...feed].reverse().slice(-40).map((r, i) => ({ i, ms: r.total_ms }));
  const avgMs = stats.n ? stats.ms / stats.n : 0;
  const flagged = stats.High + stats.Medium;

  return (
    <div className="page wide">
      <div className="page-head">
        <div>
          <h1>Live pipeline</h1>
          <p className="lede">Real weekly records from 2,000 held-out customers, scored one by one by the production model. Click a record to see every stage.</p>
        </div>
        <div className="toolbar">
          <div className="seg">
            {(Object.keys(SPEEDS) as (keyof typeof SPEEDS)[]).map((s) => (
              <button key={s} className={speed === s ? "on" : ""} onClick={() => setSpeed(s)}>{s}</button>
            ))}
          </div>
          {running
            ? <button className="btn" onClick={stop}><span className="live-dot" /> Pause</button>
            : <button className="btn primary" onClick={start}>{stats.n ? "Resume stream" : "Start live stream"}</button>}
        </div>
      </div>
      {err && <p className="small err" style={{ margin: "-10px 0 16px" }}>{err}</p>}

      <div className="overview">
        <div className="card">
          <div className="card-head"><h3>Risk tiers</h3><span className="small muted">this session</span></div>
          <div className="donut-wrap">
            <div className="donut">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={stats.n ? tiers : [{ t: "none", n: 1 }]} dataKey="n" innerRadius={58} outerRadius={82} paddingAngle={stats.n ? 3 : 0}
                    cornerRadius={6} stroke="none" startAngle={90} endAngle={-270} isAnimationActive={false}>
                    {(stats.n ? tiers : [{ t: "none" }]).map((x) => (
                      <Cell key={x.t} fill={x.t === "none" ? "var(--surface-2)" : tierColor(x.t as "High")} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
              <div className="donut-center"><div><b>{stats.n.toLocaleString()}</b><span>records</span></div></div>
            </div>
            <div className="legend">
              {tiers.map(({ t, n }) => (
                <div className="legend-row" key={t}>
                  <i style={{ background: tierColor(t) }} />
                  <span>{t} <span className="muted">({stats.n ? pct(n / stats.n, 0) : "0%"})</span></span>
                  <b>{n.toLocaleString()}</b>
                </div>
              ))}
              <div className="small muted">{stats.skip} records fired no early-warning rule.</div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="stat-label">Average latency per record</div>
          <div className="stat-value">{stats.n ? `${avgMs.toFixed(0)} ms` : "0 ms"}</div>
          <div className="trend">all 11 stages, end to end</div>
          <div className="spark">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={spark} margin={{ top: 4, right: 4, bottom: 0, left: 4 }}>
                <defs>
                  <linearGradient id="sparkFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.25} />
                    <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <YAxis hide domain={["dataMin - 5", "dataMax + 5"]} />
                {stats.n > 0 && <ReferenceLine y={avgMs} stroke="var(--text-3)" strokeDasharray="3 4" />}
                <Area dataKey="ms" type="linear" stroke="var(--accent)" strokeWidth={1.6} fill="url(#sparkFill)" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="stat-label">High tier that really defaulted</div>
          <div className="stat-value">{stats.High ? pct(stats.correctHigh / stats.High, 0) : "0%"}</div>
          <div className="trend"><b>{flagged.toLocaleString()}</b> cases sent for review</div>
          <div style={{ marginTop: 22 }}>
            <div style={{ display: "flex", justifyContent: "space-between" }} className="small">
              <span className="muted">Replay progress</span><span>Week {week + 1} of 12</span>
            </div>
            <div className="progress"><span style={{ width: `${((week + (running || stats.n ? 1 : 0)) / 12) * 100}%` }} /></div>
            <label className="check" style={{ marginTop: 16 }}>
              <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> Follow the newest record
            </label>
          </div>
        </div>
      </div>

      <div className="card stagebar" style={{ marginTop: 16 }}>
        {STAGES.map((s, i) => (
          <div key={s.id} className="stage" style={{ animationDelay: `${i * 45}ms` }} data-pulse={pulse % 2}>
            <div className="stage-dot" />
            <div className="stage-name">{s.short}</div>
            <div className="stage-ms mono">{stats.n ? `${avg(s.id).toFixed(1)} ms` : "·"}</div>
          </div>
        ))}
      </div>

      <div className="live-grid" style={{ marginTop: 16 }}>
        <div className="card" style={{ padding: 0, overflow: "hidden" }}>
          <div className="panel-head"><h3 style={{ margin: 0 }}>Incoming records</h3><span className="small muted">newest first</span></div>
          <div className="feed">
            {feed.length === 0 && <div className="empty" style={{ padding: 40 }}>Press <b>Start live stream</b> to begin scoring.</div>}
            {feed.map((r, i) => (
              <button key={`${r.customer_id}-${r.week_idx}-${i}`} className={`feed-row ${selected === r ? "sel" : ""} ${i === 0 && running ? "fresh" : ""}`} onClick={() => pick(r)}>
                <span className="mono">{r.customer_id}</span>
                <span className="muted small">W{r.week_idx + 1}</span>
                <span className="riskbar" style={{ minWidth: 60 }}><span style={{ width: `${Math.max(2, r.risk * 100)}%`, background: tierColor(r.tier) }} /></span>
                <span className="mono small" style={{ width: 46, textAlign: "right" }}>{pct(r.risk, 0)}</span>
                <TierPill tier={r.tier} />
                <span className="small muted feed-ms">{r.total_ms.toFixed(0)} ms</span>
              </button>
            ))}
          </div>
        </div>

        <div className="card trail-card">
          <div className="panel-head" style={{ padding: 0, border: "none", marginBottom: 12 }}>
            <div>
              <h3 style={{ margin: 0 }}>Record trail {trace ? <span className="muted" style={{ fontWeight: 400 }}>· <span className="mono">{trace.customer_id}</span>, week {trace.week_idx + 1}</span> : null}</h3>
              {trace && <div className="small muted">{trace.steps.length} stages · {trace.total_ms.toFixed(0)} ms in total (including the GRU explanation)</div>}
            </div>
            {trace && <button className="btn" onClick={() => nav(`/customer/${trace.customer_id}?w=${trace.week_idx}`)}>Open customer →</button>}
          </div>
          {!trace && !traceBusy && <div className="empty">Select a record to see its trail.</div>}
          {traceBusy && !trace && <Loading label="Tracing" />}
          {trace && <Trail t={trace} />}
        </div>
      </div>

      <Queue />
      <Audit />
    </div>
  );
}

function Trail({ t }: { t: Trace }) {
  return (
    <ol className="trail">
      {t.steps.map((s, i) => (
        <li key={s.id} className={`trail-step ${s.status}`}>
          <div className="trail-num">{i + 1}</div>
          <div className="trail-body">
            <div className="trail-title"><b>{s.title}</b><span className="mono small muted">{s.ms.toFixed(1)} ms</span></div>
            <div className="trail-what">{s.what}</div>
            <StepData s={s} />
          </div>
        </li>
      ))}
    </ol>
  );
}

function KV({ obj }: { obj: Record<string, unknown> }) {
  return (
    <div className="kv">
      {Object.entries(obj).map(([k, v]) => (
        <div key={k}><span className="muted">{k.replace(/_/g, " ")}</span><span className="mono">{typeof v === "number" ? fmtNum(v) : String(v)}</span></div>
      ))}
    </div>
  );
}

function StepData({ s }: { s: Step }) {
  const d = s.data;
  switch (s.id) {
    case "ingest":
      return <><KV obj={{ week: d.week, "week number": d.week_number, "weeks of history": d.weeks_of_history }} /><KV obj={d.sample} /></>;
    case "features":
      return <><KV obj={{ "tabular features": d.tabular_features, "sequence features": d.sequence_features }} /><KV obj={d.examples} /></>;
    case "prefilter":
      return (
        <div>
          <div className="small" style={{ margin: "4px 0 6px" }}>{d.fired} of {d.rules.length} rules fired → gate says <b>{d.gate}</b></div>
          <div>{d.rules.filter((r: any) => r.fired).map((r: any) => (
            <span className="chip fired" key={r.rule}>{RULE_LABELS[r.rule] ?? r.rule} <span className="mono">({fmtNum(r.value)})</span></span>
          ))}</div>
        </div>
      );
    case "lgb":
      return (
        <div>
          <div className="big-p">{pct(d.probability)} <span className="small muted">log-odds {d.log_odds}</span></div>
          <div className="small muted">baseline {d.baseline_log_odds} + Σ SHAP = {d.shap_sum_check} ✓ (equals the model's log-odds)</div>
          <div style={{ marginTop: 6 }}>{d.top_drivers.map((f: any) => (
            <div className="mini-row" key={f.label}><span>{f.label} <span className="muted mono">= {typeof f.value === "number" ? fmtNum(f.value) : f.value}</span></span>
              <span className={`mono ${f.shap > 0 ? "down" : "up"}`}>{f.shap > 0 ? "+" : ""}{f.shap}</span></div>
          ))}</div>
        </div>
      );
    case "window":
      return (
        <div>
          <span className="small">matrix {d.shape[0]} × {d.shape[1]} · </span>
          {Array.from({ length: d.padded_weeks }).map((_, i) => <span key={`p${i}`} className="wk pad">·</span>)}
          {d.real_weeks.map((w: number) => <span key={w} className="wk">W{w}</span>)}
        </div>
      );
    case "gru": {
      const max = Math.max(...d.attention.map((a: any) => a.weight), 0.0001);
      return (
        <div>
          <div className="big-p">{pct(d.probability)} <span className="small muted">log-odds {d.log_odds}</span></div>
          <div className="attn">{d.attention.map((a: any) => (
            <div key={a.week} title={`week ${a.week}: ${a.weight}`}><span style={{ height: `${(a.weight / max) * 100}%` }} /><em>W{a.week}</em></div>
          ))}</div>
        </div>
      );
    }
    case "stack":
      return (
        <div>
          <table className="eq">
            <tbody>
              {d.terms.filter((x: any) => x.x !== 0).map((x: any) => (
                <tr key={x.name}><td>{x.name}</td><td className="mono">{x.x}</td><td>×</td><td className="mono">{x.coef}</td><td>=</td><td className="mono">{x.product}</td></tr>
              ))}
              <tr><td>bias</td><td /><td /><td /><td /><td className="mono">{d.bias}</td></tr>
              <tr className="sum"><td>z (sum)</td><td /><td /><td /><td /><td className="mono">{d.z}</td></tr>
            </tbody>
          </table>
          <div className="small">σ(z) = <b>{pct(d.p_stacked)}</b> · effective weights LightGBM {d.effective_weight_lgb}, GRU {d.effective_weight_gru}</div>
        </div>
      );
    case "calibrate":
      return <div className="big-p">{pct(d.before)} <span className="muted">→</span> {pct(d.after)} <span className="small muted">({d.method})</span></div>;
    case "tier":
      return (
        <div className="small">
          {pct(d.risk)} vs Medium ≥ {pct(d.medium_threshold)} and High ≥ {pct(d.high_threshold)} → <span className={`pill ${d.tier}`}>{d.tier}</span>
        </div>
      );
    case "explain":
      return (
        <div>{d.top_reasons.map((r: any) => (
          <div className="mini-row" key={r.label}><span>{r.label} <span className="muted small">({r.sources.join(" + ")})</span></span>
            <span className={`mono ${r.effect > 0 ? "down" : "up"}`}>{r.effect > 0 ? "+" : ""}{r.effect}</span></div>
        ))}</div>
      );
    case "route":
      return (
        <div className="small">
          <b>{d.action}</b>{d.needs_human_approval ? " · sent to the approval queue below" : ""}
          {d.actual_outcome && <> · what actually happened: <b style={{ color: d.actual_outcome === "defaulted" ? "var(--high)" : "var(--low)" }}>{d.actual_outcome}</b></>}
        </div>
      );
    default:
      return <KV obj={d} />;
  }
}

function Queue() {
  const [q, setQ] = useState<{ items: Case[]; pending: number } | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [view, setView] = useState<"pending" | "reviewed">("pending");
  const load = useCallback(() => live.queue().then(setQ).catch(() => {}), []);
  useEffect(() => { load(); const t = setInterval(load, 2500); return () => clearInterval(t); }, [load]);
  const act = async (id: string, d: "approve" | "reject") => { setBusy(id); try { await live.decide(id, d); await load(); } finally { setBusy(null); } };
  const items = (q?.items ?? []).filter((c) => (view === "pending") === (c.status === "pending"));
  const reviewed = (q?.items.length ?? 0) - (q?.pending ?? 0);

  return (
    <section className="section">
      <div className="section-head">
        <div>
          <h2>Approval queue</h2>
          <p>High and Medium cases wait for a person. Approving writes the outreach note with the LLM.</p>
        </div>
        <div className="seg">
          <button className={view === "pending" ? "on" : ""} onClick={() => setView("pending")}>Pending · {q?.pending ?? 0}</button>
          <button className={view === "reviewed" ? "on" : ""} onClick={() => setView("reviewed")}>Reviewed · {reviewed}</button>
        </div>
      </div>
      {items.length === 0 ? <div className="card empty">{view === "pending" ? "No pending cases. Start the stream." : "Nothing reviewed yet. Approve or reject a pending case."}</div> : (
        <div className="grid g3">
          {(view === "reviewed" ? [...items].sort((a, b) => b.updated.localeCompare(a.updated)) : items).slice(0, 9).map((c) => (
            <div className="card case" key={c.id}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span className="mono">{c.customer_id} · W{c.week_idx + 1}</span><TierPill tier={c.tier} />
              </div>
              <div className="stat-value" style={{ fontSize: 28, marginTop: 4 }}>{pct(c.risk)}</div>
              <div className="small muted">{c.action} · {c.reasons.join(", ")}</div>
              {c.status === "pending" ? (
                <div className="toolbar" style={{ marginTop: 10 }}>
                  <button className="btn primary" disabled={busy === c.id} onClick={() => act(c.id, "approve")}>{busy === c.id ? "Writing…" : "Approve"}</button>
                  <button className="btn" disabled={busy === c.id} onClick={() => act(c.id, "reject")}>Reject</button>
                </div>
              ) : (
                <div style={{ marginTop: 10 }}>
                  <span className={`pill ${c.status === "approved" ? "Low" : "neutral"}`}>{c.status}</span>
                  {c.message && (
                    <div className="narrative" style={{ fontSize: 13.5, marginTop: 12 }}>
                      <p>{c.message.summary}</p>
                      {c.message.customer_message && <p className="quote">{c.message.customer_message}</p>}
                      {c.message.recommended_action && <p className="small"><b style={{ fontWeight: 550 }}>Suggested action:</b> {c.message.recommended_action}</p>}
                      <div className="small muted">{c.message.source === "template" ? "Template (no LLM key)" : `Written by ${c.message.source}`}</div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function Audit() {
  const [a, setA] = useState<{ entries: AuditEntry[]; total: number } | null>(null);
  useEffect(() => { const f = () => live.audit(25).then(setA).catch(() => {}); f(); const t = setInterval(f, 3000); return () => clearInterval(t); }, []);
  return (
    <section className="section">
      <div className="section-head">
        <div><h2>Audit log</h2><p>Every score and every human decision, saved to <span className="mono">logs/audit_log.jsonl</span>.</p></div>
        <span className="pill neutral">{a?.total ?? 0} entries</span>
      </div>
      <div className="card table-wrap">
        <table className="table">
          <thead><tr><th>Time</th><th>Event</th><th>Customer</th><th>Week</th><th>Detail</th></tr></thead>
          <tbody>
            {a?.entries.map((e, i) => (
              <tr key={i}>
                <td className="mono small">{e.ts}</td>
                <td>{e.type === "decision" ? <span className="pill Medium">decision</span> : <span className="pill neutral">scored</span>}</td>
                <td className="mono">{e.customer_id}</td>
                <td>W{(e.week_idx ?? 0) + 1}</td>
                <td className="small">{e.type === "decision" ? `${e.decision}${e.message_source ? ` · note by ${e.message_source}` : ""}` : `${pct(e.risk)} · ${e.tier} · ${e.action} · ${Number(e.total_ms).toFixed(0)} ms`}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {!a?.entries.length && <div className="empty">Nothing logged yet.</div>}
      </div>
    </section>
  );
}
