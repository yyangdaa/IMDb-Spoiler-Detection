# IMDb‑Spoiler‑Detection

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 Overview

With the growth of user‐generated reviews, untagged spoilers in IMDb reviews can ruin the viewing experience. This project presents an end‑to‑end pipeline to automatically identify spoiler‑laden reviews via:

- **Preprocessing & Feature Engineering**: text cleaning, TF‑IDF/BOW, LDA topics, user statistics, genre one‑hots, review timing, etc.  
- **Model Zoo**:
  - **DistilBERT + Metadata** hybrid classifier  
  - **Logistic Regression** (TF‑IDF / BOW)  
  - **XGBoost** (structured features; + LDA topics)  
  - **TextCNN** with GloVe embeddings  
- **Evaluation**: precision, recall, F1, accuracy, AUC, plus confusion matrices and SHAP analyses.

## 🚀 Quickstart

1. **Clone & Setup**  
   ```bash
   git clone https://github.com/yourusername/IMDb‑Spoiler‑Detection.git
   cd IMDb‑Spoiler‑Detection
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
