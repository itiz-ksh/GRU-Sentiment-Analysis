# GRU Sentiment Analysis — IMDB Dataset

A deep learning project that uses a **Gated Recurrent Unit (GRU)** neural network to classify IMDB movie reviews as **positive** or **negative**.

---

## Project Structure

```
.
├── dataset/
│   └── IMDB Dataset.csv
├── GRU.py
├── requirements.txt
└── README.md
```

---

## Model Architecture

| Layer       | Details                              |
|-------------|--------------------------------------|
| Embedding   | input_dim=10000, output_dim=128      |
| Dropout     | rate=0.3                             |
| GRU         | units=64, dropout=0.3, recurrent_dropout=0.3 |
| Dense       | units=1, activation=sigmoid          |

- **Loss:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Metric:** Accuracy  

---

## Dataset

[IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) — 50,000 reviews (25k positive, 25k negative).

Place the CSV at `dataset/IMDB Dataset.csv` before running.

---

## Setup & Usage

### 1. Clone / download the project

```bash
git clone https://github.com/itiz-ksh/GRU-Sentiment-Analysis
cd GRU-Sentiment-Analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run training

```bash
python GRU.py
```

This will:
- Clean and tokenize the reviews
- Train the GRU model for 5 epochs
- Plot accuracy and loss curves(Shows the Test Accuracy and Loss/ Train Accuracy and Loss)
- Print test accuracy and loss
- Run two sample predictions

---

## Preprocessing

- Lowercasing, HTML tag removal, punctuation stripping
- Tokenizer vocabulary: **10,000** words
- Sequence max length: **200** tokens (post-padding/truncation)
- Train/test split: **80% / 20%**

---

## Sample Predictions

```
Absolutely brilliant! One of the best films I have seen.
-> Positive :)  (score: 0.97)

Complete disaster. Dull plot, bad acting, total waste of time.
-> Negative :(  (score: 0.04)
```

---
