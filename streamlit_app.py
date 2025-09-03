import os
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TruthGen – Corrected News Viewer", layout="wide")

st.title("📰 TruthGen — Corrected News Viewer")
st.caption("Display the model's predictions and Gemini rewrites (or placeholders)")

DATA_PATH = Path("/mnt/data/outputs/corrections.csv")

if not DATA_PATH.exists():
    st.warning("No corrections file found at /mnt/data/outputs/corrections.csv. "
               "Run Section 9 in the notebook first.")
else:
    df = pd.read_csv(DATA_PATH)

    # Sidebar filters
    st.sidebar.header("Filters")
    pred_filter = st.sidebar.selectbox("Prediction", ["All", "Fake (0)", "Real (1)"])
    keyword = st.sidebar.text_input("Search keyword (title/text)")

    view_df = df.copy()
    if pred_filter != "All":
        label = 0 if pred_filter.startswith("Fake") else 1
        view_df = view_df[view_df["prediction"] == label]

    if keyword:
        k = keyword.lower()
        view_df = view_df[
            view_df["title"].fillna("").str.lower().str.contains(k) |
            view_df["original_text"].fillna("").str.lower().str.contains(k)
        ]

    st.sidebar.write(f"Matched rows: **{len(view_df)}**")

    if len(view_df) == 0:
        st.info("No items matched your filters.")
    else:
        titles = [f"{i+1}. {t[:90]}{'…' if isinstance(t,str) and len(t)>90 else ''}" 
                  for i, t in enumerate(view_df['title'].fillna('Untitled').tolist())]
        idx = st.selectbox("Select an article", options=list(range(len(view_df))), 
                           format_func=lambda i: titles[i])

        row = view_df.iloc[idx]

        # Display header + metadata
        cols = st.columns([3,1,1,1])
        with cols[0]:
            st.subheader(row.get("title", "Untitled"))
            meta = []
            if isinstance(row.get("subject",""), str) and row["subject"]:
                meta.append(f"**Subject:** {row['subject']}")
            if isinstance(row.get("date",""), str) and row["date"]:
                meta.append(f"**Date:** {row['date']}")
            if meta:
                st.caption(" • ".join(meta))
        with cols[1]:
            pred_text = "Fake (0)" if row["prediction"] == 0 else "Real (1)"
            st.metric("Prediction", pred_text)
        with cols[2]:
            st.metric("Has Correction", "Yes" if isinstance(row.get("corrected_text",""), str) and len(row["corrected_text"])>0 else "No")
        with cols[3]:
            st.metric("Chars (orig)", len(str(row.get("original_text",""))))

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Original (Excerpt)")
            st.write(row.get("original_text","")[:2000])
        with c2:
            st.markdown("### Corrected (Gemini)")
            st.write(row.get("corrected_text",""))

    st.markdown("---")
    st.caption("Tip: Update `/mnt/data/outputs/corrections.csv` to refresh results.")
