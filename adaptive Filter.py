import os, time, hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from vertexai.generative_models import GenerativeModel, GenerationConfig

# --- Model Setup ---
assert "vertex_ctx" in globals() and vertex_ctx.get("model"), "Vertex model not initialized"

FAST_MODELS = ["gemini-1.5-flash-002", "gemini-1.5-flash-001"]
model = next((GenerativeModel(name) for name in FAST_MODELS if GenerativeModel(name).count_tokens("ping")), None)

if model:
    print(f"⚡ Using faster model: {getattr(model, 'model_name', getattr(model, 'name', str(model)))}")
else:
    model = vertex_ctx["model"]
    print("ℹ️ Using existing model from vertex_ctx")

GEN_CFG = GenerationConfig(temperature=0.1, max_output_tokens=256)
MAX_CHARS = 1200

# --- Output Setup ---
OUT_DIR = Path("/workspaces/DS510-Team-Project/Outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "corrections.csv"

# --- Prompt Builder ---
def build_rewrite_prompt(text: str) -> str:
    return (
        "Rewrite the article truthfully and concisely, removing misinformation. "
        "If facts are uncertain, state that and suggest reliable sources. "
        "Neutral, journalistic tone. Keep it under ~6 sentences.\n\nARTICLE:\n" + (text or "")
    )

# --- Text Extraction ---
def extract_text(resp):
    try:
        parts = [
            p.text for c in getattr(resp, "candidates", []) 
            if c.content for p in getattr(c.content, "parts", []) 
            if getattr(p, "text", None)
        ]
        return "\n".join(parts) if parts else None
    except Exception:
        pass
    t = getattr(resp, "text", None)
    return t.strip() if isinstance(t, str) and t.strip() else None

# --- Hashing ---
def hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

# --- Resumable State ---
processed_idx, processed_hash = set(), set()
if OUT_CSV.exists():
    try:
        prev = pd.read_csv(OUT_CSV)
        processed_idx = set(prev.get("index", pd.Series()).dropna().astype(int))
        processed_hash = set(prev.get("body_hash", pd.Series()).dropna().astype(str))
    except Exception:
        pass

# --- Data Selection ---
assert "X_test" in globals() and "y_pred" in globals()
x_series = X_test.squeeze() if hasattr(X_test, "iloc") else pd.Series(X_test)
y_pred = np.asarray(y_pred)
fake_idx = [int(i) for i in np.where(y_pred == 0)[0]]

MAX_TO_PROCESS = None
if MAX_TO_PROCESS:
    fake_idx = fake_idx[:MAX_TO_PROCESS]

todo_idx = [i for i in fake_idx if i not in processed_idx]

# --- Adaptive Throttling ---
BATCH_SIZE_INIT = 5
PER_CALL_DELAY, BATCH_DELAY = 0.25, 1.0
RETRIES, BACKOFF = 3, 1.6
ok_streak, err_streak, batch_size = 0, 0, BATCH_SIZE_INIT

def tune_after(success: bool):
    global ok_streak, err_streak, batch_size, PER_CALL_DELAY, BATCH_DELAY
    if success:
        ok_streak += 1
        err_streak = 0
        if ok_streak >= 3:
            batch_size = min(batch_size + 1, 8)
            PER_CALL_DELAY = max(0.15, PER_CALL_DELAY * 0.9)
            BATCH_DELAY = max(0.6, BATCH_DELAY * 0.9)
            ok_streak = 0
    else:
        err_streak += 1
        ok_streak = 0
        batch_size = max(2, batch_size - 1)
        PER_CALL_DELAY = min(0.8, PER_CALL_DELAY * 1.4)
        BATCH_DELAY = min(3.0, BATCH_DELAY * 1.4)

def safe_generate(prompt: str):
    delay = 0.0
    for attempt in range(1, RETRIES + 1):
        if delay: time.sleep(delay)
        try:
            resp = model.generate_content(prompt, generation_config=GEN_CFG)
            txt = extract_text(resp) or "[Empty/blocked response]"
            tune_after(True)
            return txt
        except Exception as e:
            msg = str(e).lower()
            delay = (BACKOFF ** (attempt - 1)) + (0.5 if "429" in msg or "rate" in msg or "quota" in msg else 0)
            if attempt == RETRIES:
                tune_after(False)
                return f"[Rewrite failed after {RETRIES} retries: {e}] {prompt[:220]}{'...' if len(prompt)>220 else ''}"

# --- Rewrite Batch ---
def rewrite_with_gemini_batch(texts):
    out, i = [], 0
    while i < len(texts):
        chunk = texts[i:i+batch_size]
        for t in chunk:
            raw = str(t or "")[:MAX_CHARS]
            prompt = build_rewrite_prompt(raw)
            out.append(safe_generate(prompt))
            time.sleep(PER_CALL_DELAY)
        time.sleep(BATCH_DELAY)
        i += len(chunk)
    return out

# --- Process & Save ---
WINDOW_SIZE = 40
written_total = 0

for start in range(0, len(todo_idx), WINDOW_SIZE):
    window = todo_idx[start:start + WINDOW_SIZE]
    if not window: continue

    texts, hashes, keep_pos = [], [], []
    for i in window:
        raw = str(x_series.iloc[i])
        h = hash_text(raw)
        if h in processed_hash: continue
        texts.append(raw)
        hashes.append(h)
        keep_pos.append(i)

    if not keep_pos: continue

    rewrites = rewrite_with_gemini_batch(texts)

    df_out = pd.DataFrame({
        "index": keep_pos,
        "prediction": y_pred[keep_pos].astype(int),
        "original_excerpt": [t[:300] + ("..." if len(t) > 300 else "") for t in texts],
        "rewrite": rewrites,
        "body_hash": hashes,
    })

    df_out.to_csv(OUT_CSV, mode="a", header=not OUT_CSV.exists(), index=False, encoding="utf-8")
    processed_hash.update(hashes)
    written_total += len(df_out)
    print(f"✅ Wrote {len(df_out)} rows (total this run: {written_total}) → {OUT_CSV}")

# --- Preview ---
try:
    from IPython.display import display
    display(pd.read_csv(OUT_CSV).tail(10))
except Exception:
    print(pd.read_csv(OUT_CSV).tail(10).to_string(index=False))