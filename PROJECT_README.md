# TruthGen: AI-Powered Fake News Detection and Correction

**Authors:** Reema Ramachandra Kadechkar, Keerthi Turakapalli, Ashwin Shastry Paturi  
**Course:** DS510_03_IN: Artificial Intelligence for Data Science  
**Institution:** City University of Seattle  

---

## Project Overview

TruthGen is an end-to-end system designed to **detect** and **correct** fake news articles using a combination of **Natural Language Processing (NLP)** and **Generative AI**.

- **Phase I — Detection:**  
  Uses TF-IDF vectorization and Logistic Regression to classify news as *fake* or *real*. Achieved ~99% accuracy on Kaggle’s Fake & True News dataset.

- **Phase II — Correction:**  
  Integrates Google’s **Gemini Pro** (via Vertex AI) to rewrite fake news into factually accurate content. Prompt engineering ensures outputs are concise, truthful, and neutral in tone.

- **UI:**  
  A lightweight **Streamlit web app** allows interactive exploration of original vs. corrected articles.

---

## Repository Structure

```
/Datasets
    Fake.csv
    True.csv

/notebooks
    TruthGen_Final_Notebook.ipynb

/ui
    streamlit_app.py

/outputs
    corrections.csv (generated after running notebook)

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
Open and run `notebooks/TruthGen_Final_Notebook.ipynb` (works in Codespaces, Jupyter, or Colab).  
- Preprocesses data  
- Trains/evaluates classifier  
- Generates corrected articles  
- Exports `/outputs/corrections.csv` for UI

### 3. Launch the Streamlit UI
```bash
streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```
Open **http://localhost:8501** or the forwarded port in Codespaces.

---

## Results Summary

- **Classifier:** Logistic Regression (TF-IDF features, 1–2 n-grams)  
- **Accuracy:** ~99%  
- **Precision/Recall/F1:** Balanced across classes  
- **Correction Phase:** Fake articles rewritten with Gemini Pro into factual alternatives  
- **UI:** Side-by-side comparison of original vs. corrected text

---

## Project Timeline

| Date       | Milestone |
|------------|-----------|
| Week 1     | Dataset collection (Fake.csv, True.csv), initial EDA |
| Week 2     | Preprocessing pipeline (cleaning, stopwords, lemmatization) |
| Week 3     | Implemented TF-IDF + Logistic Regression classifier |
| Week 4     | Evaluation (accuracy, precision, recall, F1, confusion matrix) |
| Week 5     | Integrated Vertex AI (Gemini Pro) for correction |
| Week 6     | Developed Streamlit UI for visualization |
| Week 7     | Final polishing: consolidated notebook, Codespaces-ready setup, documentation |

---

## Next Steps / Future Work

- Add baselines with **SVM, Random Forest, BERT**  
- Test on external datasets (e.g., LIAR, PolitiFact) for robustness  
- Integrate **fact-check APIs** to verify Gemini rewrites  
- Extend support for **multilingual fake news detection**  
- Improve ethical safeguards (bias, hallucinations, provenance tracking)

---

## References

- Shu, K. et al. (2017). Fake news detection on social media. ACM SIGKDD Explorations Newsletter.  
- Ahmed, H. et al. (2018). Detecting opinion spams and fake news using text classification.  
- Yogiyulianto (2022). News Classification with TF-IDF & Machine Learning. Kaggle.  
- Kaliyar, R. et al. (2021). FakeBERT: Fake news detection in social media with a BERT-based approach.  
- OpenAI (2023). GPT-4 Technical Report.  
- Google Cloud (2024). Vertex AI documentation: Generative AI with Gemini Pro.

---

**End of Document**
