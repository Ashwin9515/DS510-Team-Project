## TruthGen UI Layer (`app.py` + `backend.py`)

The TruthGen UI is built with **Streamlit** and provides an interactive way to explore fake-news detection results and Gemini-powered corrections.  
It consists of two coordinated components:

---

### app.py` — Streamlit Frontend

**Purpose:**  
Acts as the main user interface, providing two key experiences:  
1. **Live Demo:** Run an on-demand rewrite of a random fake news article.  
2. **Top-N Corrections:** Showcase the most confident fake articles with their corrected versions (from `Outputs/corrections.csv`).

**Key features:**
- **Configuration:**  
  - Dataset paths: `FAKE_DATASET`, `TRUE_DATASET` (default: `Datasets/`).  
  - Corrections file: `CORR_CSV` (default: `Outputs/corrections.csv`).  
  - `UI_TOP_N`: how many rows to show in the Top-N table.  
- **Backend status indicator:**  
  - Displays whether rewrites are powered by Vertex AI, google-generativeai, or are unavailable.  
  - Controls whether the “Rewrite” button is enabled.  
- **Data loading:**  
  - Cached loading of Fake/True CSVs, with safe concatenation into a `body` column.  
- **Live Demo panel:**  
  - Left pane: select a new random fake, view title/subject/date and original text excerpt.  
  - Right pane: rewrite the sample using Gemini (if backend available) and display corrected text.  
- **Top-N Corrections panel:**  
  - Loads `corrections.csv` exported from the notebook.  
  - Normalizes `prediction` and `confidence`.  
  - Optional keyword filter on `title` and `original_excerpt`.  
  - Displays top-N rows with `index`, `prediction` (0=fake, 1=real), confidence, title, excerpt, and corrected/rewrite text.

---

### backend.py` — Rewrite Engine

**Purpose:**  
Abstracts away the details of Gemini model connectivity and provides a unified `rewrite_text()` function for the UI.

**Key features:**
- **Backend initialization (`init_rewriter`):**
  - **Vertex AI (preferred):**
    - Uses Google Application Default Credentials (ADC).
    - Reads project/region/model from environment (`VERTEX_PROJECT_ID`, `VERTEX_LOCATION`, `VERTEX_MODEL_NAME`).
    - Verifies availability via `count_tokens("ping")`.
  - **google-generativeai fallback:**
    - Uses `GEMINI_API_KEY` and `GEMINI_API_MODEL` (default: `gemini-2.5-pro`).
    - Verifies connectivity with `count_tokens("ping")`.
  - If neither path succeeds, returns a context with no rewrite backend.
- **Utilities:**
  - `build_body(title, text)`: safely concatenates title + text for preprocessing.  
  - `_extract_text(resp)`: normalizes output from both Vertex and GAI responses.  
- **`rewrite_text(text, ctx)`:**
  - Prepares a correction prompt:
    > *"Rewrite the following news article truthfully and concisely, removing any misinformation. If facts are uncertain, state uncertainty and suggest reliable sources. Keep a neutral, journalistic tone."*
  - Handles both Vertex and google-generativeai calls with retries and exponential backoff.  
  - Returns corrected text or a friendly placeholder on failure.

---

### Why This UI Design

- **Separation of concerns:**  
  - Notebook = training/evaluation + batch export (`corrections.csv`).  
  - Backend = robust, dual-path rewrite engine with fallbacks.  
  - Frontend = visualization and interactivity.
- **Demo resilience:**  
  - Live Demo gracefully disables if no backend is configured.  
  - Top-N table works offline from `corrections.csv`.  
- **Transparency:**  
  - Backend status displayed clearly.  
  - Columns normalized in-app for robust CSV handling.  
- **Extensibility:**  
  - Easy to plug in additional backends or extend UI with more filters/controls.

---

**Usage flow:**
1. Run the notebook → export `Outputs/corrections.csv`.  
2. Launch UI:  
   ```bash
   python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0