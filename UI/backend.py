# backend.py
import os, time
from typing import Optional, Dict


# Optional: use Vertex first, then google-generativeai
def init_rewriter() -> Dict:
    ctx = {"use_vertex": False, "use_gapi": False, "vertex": None, "gapi": None}

    # Try Vertex AI
    try:
        import vertexai
        from google.auth import default as google_auth_default
        from google.auth.transport.requests import Request
        from vertexai.generative_models import GenerativeModel, GenerationConfig

        creds, proj = google_auth_default()
        if hasattr(creds, "with_scopes"):
            creds = creds.with_scopes(["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(Request())

        project = os.getenv("VERTEX_PROJECT_ID") or proj
        region  = os.getenv("VERTEX_LOCATION", "us-central1")
        model_name = os.getenv("VERTEX_MODEL_NAME", "gemini-1.5-pro-002")

        vertexai.init(project=project, location=region, credentials=creds)
        mdl = GenerativeModel(model_name)
        _ = mdl.count_tokens("ping")

        ctx.update({
            "use_vertex": True,
            "vertex": {"model": mdl, "gen_cfg": GenerationConfig(temperature=0.2, max_output_tokens=768)}
        })
        return ctx
    except Exception:
        pass

    # Fallback: google-generativeai (API key required)
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            mdl = genai.GenerativeModel(os.getenv("GEMINI_API_MODEL", "gemini-2.5-pro"))
            _ = mdl.count_tokens("ping")
            ctx.update({"use_gapi": True, "gapi": {"model": mdl, "temperature": 0.2, "max_tokens": 768}})
            return ctx
    except Exception:
        pass

    # No backend — app will show “Unavailable”
    return ctx

def build_body(title_series, text_series):
    """Safely combine title + text into one string column."""
    import pandas as pd
    t = (title_series if title_series is not None else "").fillna("") if hasattr(title_series, "fillna") else ""
    x = (text_series  if text_series  is not None else "").fillna("") if hasattr(text_series,  "fillna") else ""
    if hasattr(t, "astype"):
        return (t.astype(str) + ". " + x.astype(str)).str.strip()
    return (str(t) + ". " + str(x)).strip()

def _extract_text(resp) -> Optional[str]:
    try:
        cands = getattr(resp, "candidates", None)
        if cands:
            parts = []
            for c in cands:
                content = getattr(c, "content", None)
                for p in (getattr(content, "parts", None) or []):
                    txt = getattr(p, "text", None)
                    if txt: parts.append(txt)
            if parts:
                return "\n".join(parts)
    except Exception:
        pass
    t = getattr(resp, "text", None)
    return t if isinstance(t, str) and t.strip() else None

def rewrite_text(text: str, ctx: dict) -> str:
    """Single rewrite call with gentle backoff; works for Vertex or google-generativeai."""
    text = (text or "")[:6000]
    prompt = (
        "Rewrite the following news article truthfully and concisely, removing any misinformation. "
        "If facts are uncertain, state uncertainty and suggest reliable sources. "
        "Keep a neutral, journalistic tone.\n\nARTICLE:\n" + text
    )

    for attempt in range(1, 5):
        try:
            # Vertex path
            if ctx.get("use_vertex"):
                mdl = ctx["vertex"]["model"]
                cfg = ctx["vertex"]["gen_cfg"]
                resp = mdl.generate_content(prompt, generation_config=cfg)
                out = _extract_text(resp) or "[Empty/blocked response]"
                return out
            # google-generativeai path
            if ctx.get("use_gapi"):
                mdl = ctx["gapi"]["model"]
                resp = mdl.generate_content(prompt, generation_config={"temperature": ctx["gapi"]["temperature"]})
                out = _extract_text(resp) or "[Empty/blocked response]"
                return out
            # No backend
            return "[Rewrite backend unavailable]"
        except Exception as e:
            # Basic backoff; a little extra if it smells like 429/rate
            s = str(e).lower()
            delay = (1.6 ** (attempt - 1)) + (0.5 if ("429" in s or "rate" in s or "quota" in s) else 0)
            time.sleep(delay)
    return "[Rewrite failed after retries]"
