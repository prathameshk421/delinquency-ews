import { useEffect, useState } from "react";
import { Navigate, NavLink, Route, Routes } from "react-router-dom";
import { api, Health } from "./api";
import { Logo } from "./components/ui";
import Overview from "./pages/Overview";
import Watchlist from "./pages/Watchlist";
import Customer from "./pages/Customer";
import Pipeline from "./pages/Pipeline";
import Live from "./pages/Live";

const THEMES = ["auto", "light", "dark"] as const;
const ThemeIcon = ({ t }: { t: string }) => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    {t === "dark" ? <path d="M20 14.5A8 8 0 019.5 4 8 8 0 1020 14.5z" />
      : t === "light" ? <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>
      : <><circle cx="12" cy="12" r="9" /><path d="M12 3v18" /><path d="M12 3a9 9 0 010 18z" fill="currentColor" /></>}
  </svg>
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

  const ok = !!health?.models_trained;
  const statusText = !health ? "API offline" : !ok ? "Models not trained" : health.llm.configured ? "Models · LLM ready" : "Models ready";
  const next = THEMES[(THEMES.indexOf(theme as (typeof THEMES)[number]) + 1) % THEMES.length];

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand"><Logo /><span className="brand-name">Early Signal</span></div>
        <nav className="nav">
          <NavLink to="/" end className="nav-item">Live</NavLink>
          <NavLink to="/watchlist" className="nav-item">Watchlist</NavLink>
          <NavLink to="/customer" className="nav-item">Customers</NavLink>
          <NavLink to="/report" className="nav-item">Model report</NavLink>
          <NavLink to="/pipeline" className="nav-item">Pipeline</NavLink>
        </nav>
        <div className="topbar-right">
          <span className="status" title={health?.llm.configured ? `LLM: ${health.llm.model}` : "No LLM key: template mode"}>
            <span className="status-dot" style={{ background: ok ? "var(--low)" : "var(--high)" }} />{statusText}
          </span>
          <button className="icon-btn" title={`Theme: ${theme} (click for ${next})`} onClick={() => setTheme(next)}><ThemeIcon t={theme} /></button>
        </div>
      </header>
      <main className="main">
        <Routes>
          <Route path="/" element={<Live />} />
          <Route path="/report" element={<Overview />} />
          <Route path="/watchlist" element={<Watchlist />} />
          <Route path="/portfolio" element={<Navigate to="/watchlist" replace />} />
          <Route path="/customer" element={<Customer />} />
          <Route path="/customer/:id" element={<Customer />} />
          <Route path="/pipeline" element={<Pipeline />} />
        </Routes>
      </main>
    </div>
  );
}
