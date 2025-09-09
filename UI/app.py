# app.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from backend import init_rewriter, rewrite_text, build_body

# Load environment variables
load_dotenv()

# Config
FAKE_DATASET = os.getenv("FAKE_DATASET", "/workspaces/DS510-Team-Project/Datasets/Fake.csv")
TRUE_DATASET = os.getenv("TRUE_DATASET", "/workspaces/DS510-Team-Project/Datasets/True.csv")
CORR_CSV     = Path("/workspaces/DS510-Team-Project/Outputs/corrections.csv")
TOP_N        = int(os.getenv("UI_TOP_N", "20"))
MAX_SHOW     = 2000  # chars to preview from original

# UI Setup
st.set_page_config(page_title="TruthGen Demo", layout="wide")
st.title("📰 TruthGen: AI-Powered Fake News Detection and Correction")

# Backend Status
ctx = init_rewriter()
status = "Vertex" if ctx.get("use_vertex") else ("google-generativeai" if ctx.get("use_gapi") else "Unavailable")
st.caption(f"Rewrite backend: **{status}**")

# Data Loading
@st.cache_data(show_spinner=False)
def load_df(fake_path: str, true_path: str):
    fp, tp = Path(fake_path), Path(true_path)
    if not (fp.exists() and tp.exists()):
        return None
    fake = pd.read_csv(fp); fake["label"] = 0
    true = pd.read_csv(tp); true["label"] = 1
    df = pd.concat([fake, true], ignore_index=True)
    df["body"] = build_body(df.get("title"), df.get("text"))
    return df

df = load_df(FAKE_DATASET, TRUE_DATASET)

# Live Demo Section
st.subheader("Live Demo")
left, right = st.columns([3, 2])

def pick_random_fake(d: pd.DataFrame):
    if d is None or "label" not in d: return None
    fakes = d[d["label"] == 0]
    if fakes.empty: return None
    return fakes.sample(1).iloc[0]

if "rand_row" not in st.session_state:
    st.session_state.rand_row = pick_random_fake(df)
if "last_rewrite" not in st.session_state:
    st.session_state.last_rewrite = None

with left:
    if st.button("New random fake", use_container_width=True):
        st.session_state.rand_row = pick_random_fake(df)
        st.session_state.last_rewrite = None

    row = st.session_state.rand_row
    if row is None:
        st.info("Dataset not found or has no label==0 rows.")
    else:
        st.markdown(f"**Title:** {str(row.get('title') or 'Untitled')[:200]}")
        st.caption(f"Subject: {row.get('subject','') or '—'} • Date: {row.get('date','') or '—'}")
        st.markdown("**Original (excerpt)**")
        st.write((row.get("body") or row.get("text") or "")[:MAX_SHOW])

with right:
    if st.session_state.rand_row is not None:
        if st.button("Rewrite using Gemini", use_container_width=True, type="primary", disabled=(status == "Unavailable")):
            with st.spinner("Rewriting..."):
                try:
                    text_in = (row.get("body") or row.get("text") or "")[:6000]
                    st.session_state.last_rewrite = rewrite_text(text_in, ctx=ctx)
                except Exception as e:
                    st.session_state.last_rewrite = f"[Rewrite failed: {e}]"
        st.markdown("**Corrected (Gemini)**")
        st.write(st.session_state.last_rewrite or "Click **Rewrite using Gemini** to generate a correction.")

st.markdown("---")

# Top-N Corrections Section
st.subheader("Top 20 Most Confident Corrections")
if not CORR_CSV.exists():
    st.caption("No corrections file yet. Generate it from the notebook (Top-50 export).")
else:
    corr = pd.read_csv(CORR_CSV)

    # Normalize optional columns
    if "prediction" in corr.columns:
        corr["prediction"] = pd.to_numeric(corr["prediction"], errors="coerce").fillna(-1).astype(int)
    if "confidence" in corr.columns:
        corr["confidence"] = pd.to_numeric(corr["confidence"], errors="coerce")

    # Sort by confidence if available
    view = corr.copy()
    if "confidence" in view.columns:
        view = view.sort_values("confidence", ascending=False)

    q = st.text_input("Filter (title / original excerpt)", placeholder="e.g., election, vaccine")
    if q:
        ql = q.lower()
        def _contains(s):
            s = "" if not isinstance(s, str) else s
            return ql in s.lower()
        view = view[
            view.get("title","").apply(_contains) |
            view.get("original_excerpt","").apply(_contains)
        ]

    # Derive display columns robustly
    pred_text = np.where(view.get("prediction", pd.Series([-1]*len(view))) == 0, "Fake (0)", "Real (1)")
    table = pd.DataFrame({
        "index": view.get("index", range(len(view))),
        "pred": pred_text,
        **({"confidence": view["confidence"]} if "confidence" in view.columns else {}),
        "title": view.get("title", ""),
        "original_excerpt": view.get("original_excerpt", ""),
        "rewrite": view.get("rewrite", view.get("corrected_text", "")),
    })

    st.dataframe(table.head(TOP_N), use_container_width=True, hide_index=True)

st.caption("Tip: Re-run or click “New random fake” for another sample.")