"""Plain-language risk narrative via Groq (OpenAI-compatible endpoint).

Configure in .env:
    GROQ_API_KEY=...
    GROQ_MODEL=openai/gpt-oss-20b

If the key is missing or the call fails (model retired, rate limit, network),
a deterministic template narrative is returned instead so the app never breaks.
"""
import json
import os
import re

import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"

SYSTEM = (
    "You are a credit-risk analyst assistant at a retail bank. You turn model outputs into short, "
    "factual explanations for a collections/retention officer. Rules: only use the numbers you are "
    "given; never invent customer facts; never promise waivers, settlements or restructuring; the "
    "customer message must be supportive, non-threatening and must not mention scores or models. "
    "Reply with a single JSON object and nothing else."
)


def config():
    return {"configured": bool(os.getenv("GROQ_API_KEY")), "model": os.getenv("GROQ_MODEL", DEFAULT_MODEL)}


def _prompt(s):
    factors = [f"- {f['label']}: value {f['value']}, {'raises' if f['contribution'] > 0 else 'lowers'} risk"
               for f in s["factors"]["combined"][:6]]
    return (
        f"Customer {s['customer_id']}, observation week {s['week_idx'] + 1} of history.\n"
        f"Calibrated probability of default in the next 2-4 weeks: {s['risk']:.2f} (tier: {s['tier']}).\n"
        f"Tabular model probability {s['p_lgb']:.2f}; behavioural-sequence model probability {s['p_gru']:.2f}.\n"
        f"Weeks that most influenced the sequence model: {', '.join(str(w['week']) for w in s['week_importance'][:3])}.\n"
        f"Early-warning rules fired: {', '.join(s['prefilter']['rules_fired']) or 'none'}.\n"
        f"Top drivers:\n" + "\n".join(factors) + "\n\n"
        'Return JSON with keys: "summary" (2 sentences), "reason_codes" (3-4 short phrases), '
        '"recommended_action" (one of: monitor, digital nudge, relationship-manager call, '
        'financial counselling referral; plus a one-line why), "customer_message" (<= 60 words).'
    )


def template(s):
    top = [f["label"] for f in s["factors"]["combined"] if f["contribution"] > 0][:3]
    action = {"High": "relationship-manager call", "Medium": "digital nudge"}.get(s["tier"], "monitor")
    return {
        "summary": (f"Estimated {s['risk']:.0%} chance of default in the next 2-4 weeks ({s['tier']} risk). "
                    f"Main drivers: {', '.join(top) or 'no strong stress signals'}."),
        "reason_codes": top or ["No dominant stress signal"],
        "recommended_action": f"{action} - based on the {s['tier'].lower()} risk tier.",
        "customer_message": ("Hi, we noticed a few changes in your account recently and wanted to check in. "
                             "If your upcoming EMI might be difficult, reply to this message and our team "
                             "will walk you through the options available."),
        "source": "template",
    }


def narrate(scored: dict) -> dict:
    cfg = config()
    if not cfg["configured"]:
        return {**template(scored), "note": "GROQ_API_KEY not set - showing template narrative."}
    body = {
        "model": cfg["model"],
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": _prompt(scored)}],
        "temperature": 0.2,
        "max_completion_tokens": 800,
    }
    if cfg["model"].startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"
    try:
        r = httpx.post(GROQ_URL, json=body, timeout=30,
                       headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"})
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"] or ""
        m = re.search(r"\{.*\}", text, re.S)
        out = json.loads(m.group(0)) if m else {"summary": text.strip()}
        out.setdefault("reason_codes", [])
        out["source"] = cfg["model"]
        return out
    except Exception as e:  # noqa: BLE001 - any failure falls back to the template
        detail = str(e)
        if isinstance(e, httpx.HTTPStatusError):
            detail = f"{e.response.status_code}: {e.response.text[:200]}"
        return {**template(scored), "note": f"Groq call failed ({detail}) - showing template narrative."}
