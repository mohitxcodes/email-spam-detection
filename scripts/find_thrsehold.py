# scripts/find_threshold.py
import joblib
import numpy as np
from sklearn.metrics import precision_recall_curve
import pandas as pd
from pathlib import Path

MODEL_P = Path("models/spam_model.joblib")
VECT_P  = Path("models/vectorizer.joblib")
TEST_P  = Path("/Users/mohitsingh/code/email-spam/data/cleaned_emails.csv")

model = joblib.load(MODEL_P)
vectorizer = joblib.load(VECT_P)

df = pd.read_csv(TEST_P)

# Ensure cleaned_message exists
if 'cleaned_message' not in df.columns:
    raise KeyError("'cleaned_message' column not found in test.csv. Run cleaning script first.")

# Ensure label numeric and drop NaNs
if 'label' not in df.columns:
    raise KeyError("'label' column not found in test.csv. Fix labels first.")

n_na = df['label'].isna().sum()
if n_na:
    print(f"Warning: {n_na} NaN labels found. These rows will be dropped before analysis.")
    df = df[~df['label'].isna()].copy()

# Force int conversion (safe now)
df['label'] = df['label'].astype(int)

X_test = df['cleaned_message']
y_test = df['label']

X_test_tfidf = vectorizer.transform(X_test)
y_prob = model.predict_proba(X_test_tfidf)[:,1]

prec, rec, thr = precision_recall_curve(y_test, y_prob)

print("Showing some candidate thresholds (precision, recall, threshold):")
for p, r, t in zip(prec[::10], rec[::10], thr[::10]):
    print(f"precision={p:.3f}, recall={r:.3f}, threshold={t:.3f}")

# optionally compute threshold that maximizes F1
from sklearn.metrics import f1_score
best_f1 = 0
best_t = 0.5
for t in np.linspace(0.01, 0.99, 99):
    preds = (y_prob >= t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t
print(f"\nBest F1={best_f1:.3f} at threshold={best_t:.2f}")