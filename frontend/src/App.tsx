import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { api, Health } from "./api";
import { Logo } from "./components/ui";
import Overview from "./pages/Overview";
import Portfolio from "./pages/Portfolio";
import Customer from "./pages/Customer";
import Pipeline from "./pages/Pipeline";
import Live from "./pages/Live";

const Icon = ({ d }: { d: string }) => (
  <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d={d} /></svg>
);

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [theme, setTheme] = useState<string>(() => {
    try { return localStorage.getItem("theme") || "auto"; } catch { return "auto"; }
  });

  useEffect(() => { api.health().then(setHealth).catch(() => setHealth(null)); }, []);
  useEffect(() => {
    if (theme === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("theme", theme); } catch { /* ignore */ }
  }, [theme]);

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand"><Logo /><span className="brand-name">Early Signal</span></div>
        <NavLink to="/" end className="nav-item"><Icon d="M3 12h4l3-8 4 16 3-8h4" />Live pipeline</NavLink>
        <NavLink to="/report" className="nav-item"><Icon d="M4 20V10M10 20V4M16 20v-7M22 20H2" />Model report</NavLink>
        <NavLink to="/portfolio" className="nav-item"><Icon d="M4 6h16M4 12h16M4 18h10" />Portfolio</NavLink>
        <NavLink to="/customer" className="nav-item"><Icon d="M12 12a4 4 0 100-8 4 4 0 000 8zM4 21a8 8 0 0116 0" />Customer analysis</NavLink>
        <NavLink to="/pipeline" className="nav-item"><Icon d="M4 7h6v4H4zM14 13h6v4h-6zM10 9h2a2 2 0 012 2v4" />Pipeline & ablations</NavLink>
        <div className="sidebar-foot">
          <div>
            <span className="status-dot" style={{ background: health?.models_trained ? "var(--low)" : "var(--high)" }} />
            {health ? (health.models_trained ? "Models loaded" : "Models not trained") : "API offline"}
          </div>
          <div>
            <span className="status-dot" style={{ background: health?.llm.configured ? "var(--low)" : "var(--text-3)" }} />
            {health?.llm.configured ? `LLM · ${health.llm.model}` : "LLM key not set (template mode)"}
          </div>
          <div className="seg" style={{ marginTop: 6 }}>
            {["auto", "light", "dark"].map((t) => (
              <button key={t} className={theme === t ? "on" : ""} onClick={() => setTheme(t)}>{t}</button>
            ))}
          </div>
          <div>Group TY_11-22 · PICT</div>
        </div>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Live />} />
          <Route path="/report" element={<Overview />} />
          <Route path="/portfolio" element={<Portfolio />} />
          <Route path="/customer" element={<Customer />} />
          <Route path="/customer/:id" element={<Customer />} />
          <Route path="/pipeline" element={<Pipeline />} />
        </Routes>
      </main>
    </div>
  );
}
