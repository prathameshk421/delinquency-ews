import { ReactNode } from "react";
import { Tier, tierColor, Factor } from "../api";

export function Logo({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
      <rect width="32" height="32" rx="9" fill="var(--accent)" />
      <path d="M6 18h5l3-8 4 13 3-7h5" stroke="#fff" strokeWidth="2.4" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

export const TierPill = ({ tier }: { tier: Tier }) => <span className={`pill ${tier}`}>{tier}</span>;

export function RiskBar({ value, tier }: { value: number; tier: Tier }) {
  return (
    <div className="riskbar">
      <span style={{ width: `${Math.max(2, value * 100)}%`, background: tierColor(tier) }} />
    </div>
  );
}

export function Section({ title, sub, right, children }: { title: string; sub?: string; right?: ReactNode; children: ReactNode }) {
  return (
    <section className="section">
      <div className="section-head">
        <div>
          <h2>{title}</h2>
          {sub && <p>{sub}</p>}
        </div>
        {right}
      </div>
      {children}
    </section>
  );
}

export function FactorList({ factors, scale }: { factors: Factor[]; scale?: number }) {
  const max = scale ?? Math.max(0.0001, ...factors.map((f) => Math.abs(f.contribution)));
  return (
    <div>
      {factors.map((f) => {
        const w = (Math.abs(f.contribution) / max) * 50;
        const up = f.contribution > 0;
        return (
          <div className="factor" key={f.feature}>
            <div>
              <div className="factor-name">{f.label}</div>
              <div className="factor-meta">
                value {typeof f.value === "number" ? fmtNum(f.value) : f.value}
                {f.sources ? ` · ${f.sources.join(" + ")}` : ""}
              </div>
            </div>
            <div className="fbar">
              <div className="axis" />
              <div className="bar" style={{ left: up ? "50%" : `${50 - w}%`, width: `${w}%`, background: up ? "var(--high)" : "var(--low)" }} />
            </div>
            <div className={`small mono ${up ? "down" : "up"}`} style={{ textAlign: "right" }}>
              {up ? "+" : "−"}{Math.abs(f.contribution).toFixed(2)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function fmtNum(v: number) {
  if (Math.abs(v) >= 1000) return v.toLocaleString("en-IN", { maximumFractionDigits: 0 });
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(2);
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return <div className="empty"><span className="typing"><span /><span /><span /></span> {label}</div>;
}

export const chartTooltip = {
  contentStyle: { background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12.5 },
  labelStyle: { color: "var(--text-2)" },
};
