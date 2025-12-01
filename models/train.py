# models/train.py
import pandas as pd
import re
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "cleaned_emails.csv"
# if you saved raw and not cleaned, point to raw:
# DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw_emails.csv"

def clean_text(text):
    if not isinstance(text, str):
        return ""
    s = text.lower()
    s = re.sub(r"\r|\n", " ", s)
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"http\S+|www\.\S+", " URL ", s)
    s = re.sub(r"\S+@\S+", " EMAIL ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

print("Initial shape:", df.shape)
print("Columns:", df.columns.tolist())

# --- DEBUG: show unique category values so we can see unexpected ones ---
if 'Category' in df.columns:
    print("\nUnique values in 'Category' (raw):")
    print(df['Category'].value_counts(dropna=False))
else:
    print("No 'Category' column found. Looking for 'label' column instead.")
    if 'label' in df.columns:
        print("Found 'label' column. Unique values:")
        print(df['label'].value_counts(dropna=False))
    else:
        raise KeyError("Neither 'Category' nor 'label' columns found in the CSV.")

# --- Normalize category column (strip + lowercase) then map to 0/1 ---
# This handles 'Spam', ' SPAM ', 'ham', 'Ham', etc.
df['category_norm'] = df['Category'].astype(str).str.strip().str.lower().replace({'nan': None})

print("\nUnique values after normalization:")
print(df['category_norm'].value_counts(dropna=False))

# Map known labels
mapping = {'spam': 1, 'ham': 0}
df['label'] = df['category_norm'].map(mapping)

# Show rows that failed mapping (label is NaN)
bad = df[df['label'].isna()]
if not bad.empty:
    print(f"\nWarning: {len(bad)} rows could not be mapped to 'spam' or 'ham'.")
    print("Examples of problematic category values:")
    print(bad['Category'].value_counts().head(10))
    # You can inspect these rows if you want:
    # print(bad.head(10))
    # For now we will drop them:
    df = df[~df['label'].isna()].copy()
    print(f"Dropped rows with unmapped categories. New shape: {df.shape}")
else:
    print("\nAll category values mapped successfully.")

# If your file already has 'cleaned_message', use it; otherwise create it
if 'cleaned_message' not in df.columns:
    print("No 'cleaned_message' column found. Creating from 'message' by applying clean_text().")
    df['cleaned_message'] = df['message'].fillna("").apply(clean_text)
else:
    # still ensure it's a string and cleaned
    df['cleaned_message'] = df['cleaned_message'].fillna("").astype(str)

# Drop empty messages
before = len(df)
df = df[df['cleaned_message'].str.strip() != ""].copy()
dropped = before - len(df)
if dropped:
    print(f"Dropped {dropped} empty messages. New shape: {df.shape}")

# Final label distribution
print("\nFinal label distribution:")
print(df['label'].value_counts())

# Prepare X, y
X = df['cleaned_message']
y = df['label'].astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# Vectorize + train
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Eval
y_pred = model.predict(X_test_tfidf)
print("\nClassification report (test):\n")
print(classification_report(y_test, y_pred))

# Save model + vectorizer
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODELS_DIR / "spam_model.joblib")
joblib.dump(vectorizer, MODELS_DIR / "vectorizer.joblib")
print(f"\nSaved model and vectorizer to: {MODELS_DIR}")