"""FastAPI service for the delinquency early-warning system.

    uvicorn api.main:app --reload --port 8000
"""
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from ews import config as C  # noqa: E402
from ews import llm  # noqa: E402
from ews.inference import Scorer  # noqa: E402

app = FastAPI(title="Pre-Delinquency Early Warning API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

scorer: Optional[Scorer] = None


def get_scorer() -> Scorer:
    global scorer
    if scorer is None:
        if not (C.ARTIFACT_DIR / "bundle.joblib").exists():
            raise HTTPException(503, "Models not trained yet. Run: python scripts/train.py")
        scorer = Scorer()
    return scorer


class ScoreRequest(BaseModel):
    customer_id: str
    week_idx: Optional[int] = None
    overrides: dict = {}


@app.get("/api/health")
def health():
    trained = (C.ARTIFACT_DIR / "bundle.joblib").exists()
    return {"status": "ok", "models_trained": trained, "llm": llm.config()}


@app.get("/api/metrics")
def metrics():
    return get_scorer().metrics


@app.get("/api/portfolio")
def portfolio(week_idx: Optional[int] = None, tier: Optional[str] = None, q: Optional[str] = None,
              limit: int = 200):
    s = get_scorer()
    p = s.portfolio
    w = int(p[C.WEEK_IDX].max()) if week_idx is None else int(week_idx)
    cur = p[p[C.WEEK_IDX] == w].copy()
    prev = p[p[C.WEEK_IDX] == w - 1].set_index(C.ID)["risk"]
    cur["delta"] = cur[C.ID].map(prev).fillna(cur["risk"]) * -1 + cur["risk"]
    counts = cur["tier"].value_counts().to_dict()
    if tier:
        cur = cur[cur["tier"] == tier]
    if q:
        cur = cur[cur[C.ID].str.contains(q.upper())]
    cur = cur.sort_values("risk", ascending=False)
    return {
        "week_idx": w,
        "weeks_available": sorted(p[C.WEEK_IDX].unique().tolist()),
        "counts": {k: int(counts.get(k, 0)) for k in ("High", "Medium", "Low")},
        "total": int(len(p[p[C.WEEK_IDX] == w])),
        "rows": cur.head(limit).round(4).to_dict(orient="records"),
    }


@app.get("/api/customers/{customer_id}")
def customer(customer_id: str):
    s = get_scorer()
    try:
        out = s.customer(customer_id.upper())
    except KeyError:
        raise HTTPException(404, f"Unknown customer {customer_id}")
    if not out["is_test_customer"]:  # not pre-scored: score each week on the fly
        for row in out["timeline"]:
            r = s.score(out["customer_id"], row["week_idx"], explain_seq=False)
            row.update({"risk": r["risk"], "p_lgb": r["p_lgb"], "p_gru": r["p_gru"]})
    return out


@app.post("/api/score")
def score(req: ScoreRequest):
    try:
        return get_scorer().score(req.customer_id.upper(), req.week_idx, req.overrides or None)
    except KeyError:
        raise HTTPException(404, f"Unknown customer {req.customer_id}")


@app.post("/api/narrative")
def narrative(req: ScoreRequest):
    try:
        scored = get_scorer().score(req.customer_id.upper(), req.week_idx, req.overrides or None)
    except KeyError:
        raise HTTPException(404, f"Unknown customer {req.customer_id}")
    return llm.narrate(scored)


# Serve the built frontend (frontend/dist) from the same port if it exists.
DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
