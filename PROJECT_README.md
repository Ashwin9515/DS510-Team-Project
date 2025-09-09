# TruthGen: AI-Powered Fake News Detection and Correction

**Authors:** Reema Ramachandra Kadechkar, Keerthi Turakapalli, Ashwin Shastry Paturi  
**Course:** DS510_03_IN: Artificial Intelligence for Data Science  
**Institution:** City University of Seattle  

---

## Project Overview

TruthGen is an end-to-end system designed to **detect** and **correct** fake news articles using a combination of **Natural Language Processing (NLP)**, **Machine Learning**, and **Generative AI**.

- **Phase I — Detection:**  
  - Preprocesses news text (stopwords, lemmatization, normalization).  
  - Uses **TF-IDF vectorization** and **Logistic Regression** (with baselines including Linear SVC) to classify articles as *fake* or *real*.  
  - Achieved **~99% accuracy** on the Kaggle Fake & True News dataset.

- **Phase II — Correction:**  
  - Integrates Google’s **Gemini (Vertex AI)**, with a fallback to the **google-generativeai SDK**, to rewrite fake news into factually accurate alternatives.  
  - Prompt design enforces outputs that are **concise, truthful, and neutral in tone**.  
  - Includes safeguards: retries, exponential backoff, and graceful fallbacks.

- **UI:**  
  - A **Streamlit web application** enables interactive exploration:  
    - Live demo on random fake articles.  
    - Side-by-side comparison of **original vs. corrected** news.  
    - Top-N table view sourced from exported corrections.

---

## Repository Structure

```
/Datasets
    Fake.csv
    True.csv

/notebooks
    TruthGen_Final_Notebook.ipynb

/ui
    app.py
    backend.py

/outputs
    corrections.csv  (generated after running notebook)
    tfidf_logreg_pipeline.joblib

requirements.txt
CODESPACES_README.md
PROJECT_README.md
```

---

## Setup Instructions

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
Open and run `notebooks/TruthGen_Final_Notebook.ipynb` in **Codespaces**, **Jupyter**, or **Colab**.  
This notebook will:  
- Preprocess the dataset  
- Train & evaluate classifiers  
- Generate corrections with Gemini (if backend configured)  
- Export `/outputs/corrections.csv` for UI consumption

### 3. Launch the Streamlit UI
```bash
streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
```
Open **http://localhost:8501** or the forwarded port in Codespaces.

---

## Results Summary

- **Classifier:** Logistic Regression (TF-IDF features, 1–2 n-grams)  
- **Accuracy:** ~99% on Kaggle dataset  
- **Precision/Recall/F1:** Balanced across classes  
- **Confusion Matrix:** Shows high separability of fake vs. real  
- **Correction Phase:** Fake articles rewritten by Gemini into factual alternatives  
- **UI:** Enables interactive exploration and validation of results

---

## Project Timeline

| Week | Milestone |
|------|-----------|
| 1    | Dataset collection (Fake.csv, True.csv), initial EDA |
| 2    | Preprocessing pipeline (cleaning, stopwords, lemmatization) |
| 3    | Implemented TF-IDF + Logistic Regression classifier |
| 4    | Evaluation (accuracy, precision, recall, F1, confusion matrix) |
| 5    | Integrated Vertex AI (Gemini) for correction |
| 6    | Developed Streamlit UI for visualization |
| 7    | Final polishing: consolidated notebook, Codespaces-ready setup, documentation |

---

## Next Steps / Future Work

- Extend baselines with **SVM, Random Forest, and BERT**  
- Validate on **external datasets** (e.g., LIAR, PolitiFact) for robustness  
- Integrate **fact-check APIs** to verify Gemini rewrites  
- Add **multilingual support** for fake news detection in non-English contexts  
- Improve **ethical safeguards** (bias mitigation, hallucination checks, provenance tracking)

---

## References

- Shu, K. et al. (2017). *Fake news detection on social media.* ACM SIGKDD Explorations Newsletter.  
- Ahmed, H. et al. (2018). *Detecting opinion spams and fake news using text classification.*  
- Yogiyulianto (2022). *News Classification with TF-IDF & Machine Learning.* Kaggle.  
- Kaliyar, R. et al. (2021). *FakeBERT: Fake news detection in social media with a BERT-based approach.*  
- OpenAI (2023). *GPT-4 Technical Report.*  
- Google Cloud (2024). *Vertex AI documentation: Generative AI with Gemini.*