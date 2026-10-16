"""FastAPI service for the delinquency early-warning system.

    uvicorn api.main:app --reload --port 8000
"""
import asyncio
import json
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
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


# --------------------------------------------------------------------------
# Live pipeline: replay held-out weekly records through the real scorer.
# --------------------------------------------------------------------------
LOG_DIR = ROOT / "logs"
AUDIT = LOG_DIR / "audit_log.jsonl"
_lock = threading.Lock()
QUEUE: dict = {}  # case_id -> case


def audit(entry: dict):
    LOG_DIR.mkdir(exist_ok=True)
    entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **entry}
    with _lock, AUDIT.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def summarize(r: dict) -> dict:
    return {
        "customer_id": r["customer_id"], "week_idx": r["week_idx"], "week": r["week"],
        "risk": r["risk"], "tier": r["tier"], "p_lgb": r["p_lgb"], "p_gru": r["p_gru"],
        "rules_fired": len(r["prefilter"]["rules_fired"]), "action": r["action"],
        "actual_label": r["actual_label"], "total_ms": r["total_ms"],
        "stages": [{"id": s["id"], "title": s["title"], "ms": s["ms"], "status": s["status"]} for s in r["steps"]],
        "reasons": [c["label"] for c in r["factors"]["combined"] if c["contribution"] > 0][:3],
    }


def enqueue(r: dict):
    if r["tier"] == "Low":
        return None
    with _lock:
        for c in QUEUE.values():  # one open case per customer
            if c["customer_id"] == r["customer_id"] and c["status"] == "pending":
                c.update(week_idx=r["week_idx"], week=r["week"], risk=r["risk"], tier=r["tier"],
                         reasons=summarize(r)["reasons"], updated=time.strftime("%H:%M:%S"))
                return c["id"]
        cid = uuid.uuid4().hex[:8]
        QUEUE[cid] = {"id": cid, "customer_id": r["customer_id"], "week_idx": r["week_idx"], "week": r["week"],
                      "risk": r["risk"], "tier": r["tier"], "action": r["action"], "status": "pending",
                      "reasons": summarize(r)["reasons"], "created": time.strftime("%H:%M:%S"),
                      "updated": time.strftime("%H:%M:%S"), "message": None}
        return cid


@app.get("/api/stream")
async def stream(request: Request, week_idx: int = 0, delay_ms: int = 700, limit_per_week: int = 60):
    """Server-sent events: each event is one real record scored by the real pipeline."""
    s = get_scorer()
    customers = list(s.b["test_customers"])

    async def gen():
        for w in range(int(week_idx), s.b["seq_len"]):
            yield f"event: week\ndata: {json.dumps({'week_idx': w})}\n\n"
            rng_order = sorted(customers, key=lambda c: hash((c, w)) % 100003)[:limit_per_week]
            for cust in rng_order:
                if await request.is_disconnected():
                    return
                r = await asyncio.to_thread(s.score, cust, w, None, False)
                case = enqueue(r)
                ev = {**summarize(r), "case_id": case}
                audit({"type": "scored", **{k: ev[k] for k in ("customer_id", "week_idx", "risk", "tier",
                                                                "action", "total_ms")}})
                yield f"event: record\ndata: {json.dumps(ev)}\n\n"
                await asyncio.sleep(max(0, delay_ms) / 1000)
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/trace")
def trace(req: ScoreRequest):
    """Full trail for one record, including the GRU's Integrated-Gradients explanation."""
    try:
        return get_scorer().score(req.customer_id.upper(), req.week_idx, req.overrides or None)
    except KeyError:
        raise HTTPException(404, f"Unknown customer {req.customer_id}")


@app.get("/api/queue")
def queue():
    with _lock:
        items = sorted(QUEUE.values(), key=lambda c: (c["status"] != "pending", -c["risk"]))
    return {"items": items[:100], "pending": sum(c["status"] == "pending" for c in items)}


class Decision(BaseModel):
    decision: str  # approve | reject
    note: str = ""


@app.post("/api/queue/{case_id}")
def decide(case_id: str, d: Decision):
    if case_id not in QUEUE:
        raise HTTPException(404, "Unknown case")
    case = QUEUE[case_id]
    if case["status"] != "pending":
        raise HTTPException(409, f"Case already {case['status']}")
    if d.decision not in ("approve", "reject"):
        raise HTTPException(400, "decision must be approve or reject")
    msg = None
    if d.decision == "approve":
        scored = get_scorer().score(case["customer_id"], case["week_idx"])
        msg = llm.narrate(scored)
    with _lock:
        case.update(status="approved" if d.decision == "approve" else "rejected", message=msg,
                    note=d.note, updated=time.strftime("%H:%M:%S"))
    audit({"type": "decision", "case_id": case_id, "customer_id": case["customer_id"],
           "week_idx": case["week_idx"], "decision": case["status"], "note": d.note,
           "message_source": msg.get("source") if msg else None})
    return case


@app.get("/api/audit")
def audit_log(limit: int = 50):
    if not AUDIT.exists():
        return {"entries": [], "total": 0}
    lines = AUDIT.read_text().splitlines()
    return {"entries": [json.loads(x) for x in lines[-limit:]][::-1], "total": len(lines)}


# Serve the built frontend (frontend/dist) from the same port if it exists.
DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
