
# IMDb‑Spoiler‑Detection

**Automatically detect untagged spoilers in IMDb user reviews**  
Using both classical ML (TF‑IDF + Logistic Regression / XGBoost) and a Transformer (DistilBERT) approach.

---

## 📁 Repository Structure

```

.
├── IMDb\_movie\_details.json   # Raw review + metadata dump from IMDb
├── data\_preprocessing.py     # Clean & feature‑engineer script
├── basic-model.ipynb         # Baseline models (TF‑IDF + LR / XGBoost)
├── transformer.ipynb         # Transformer‑based classifier (DistilBERT)
└── README.md                 # You are here

````

---

## ⚙️ Setup

1. **Clone & install**  
   ```bash
   git clone https://github.com/yourusername/IMDb‑Spoiler‑Detection.git
   cd IMDb‑Spoiler‑Detection
   pip install pandas scikit-learn xgboost transformers torch jupyter matplotlib


2. **Create a data folder**

   ```bash
   mkdir data results
   ```

---

## 🧹 1. Data Preprocessing

Run the Python script to:

* Clean & normalize review text
* Compute TF‑IDF vectors
* Engineer simple metadata features (review length, genre, timestamp, etc.)
* Output a single CSV ready for modeling

```bash
python data_preprocessing.py \
  --input IMDb_movie_details.json \
  --output data/processed_reviews.csv
```

---

## 🤖 2. Baseline Models

Open the baseline notebook:

```bash
jupyter notebook basic-model.ipynb
```

Inside you will:

1. Load `data/processed_reviews.csv`
2. Split into train / test sets
3. Train & evaluate:

   * Logistic Regression (TF‑IDF)
   * XGBoost (TF‑IDF + metadata)
4. View metrics (accuracy, precision, recall, F1) and confusion matrices
5. Export results to `results/baseline/`

---

## 🔥 3. Transformer Model

Launch the Transformer notebook:

```bash
jupyter notebook transformer.ipynb
```

It walks through:

1. Tokenizing reviews for DistilBERT
2. Fine‑tuning on spoiler classification
3. Evaluating against your baseline
4. Plotting training curves & ROC

Results will be saved under `results/transformer/`.

---

## 📊 Viewing Results

After running both notebooks, check:

```
results/
├── baseline/
│   ├── metrics.json
│   └── confusion_matrix.png
└── transformer/
    ├── metrics.json
    └── training_loss.png
```

---

## 📝 License

This project is MIT‑licensed. See [LICENSE](LICENSE) for details.

```

Feel free to adjust the `--output` path in `data_preprocessing.py` or notebook cells if your scripts save to a different location.
```
