# src/api.py
import re
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]  # project root when placed in src/
MODEL_PATH = APP_ROOT / "models" / "spam_model.joblib"
VECT_PATH  = APP_ROOT / "models" / "vectorizer.joblib"

# Load resources
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

# Set threshold you picked
THRESHOLD = 0.5

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text.lower()
    s = re.sub(r"\r|\n", " ", s)
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"http\S+|www\.\S+", " URL ", s)
    s = re.sub(r"\S+@\S+", " EMAIL ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

app = FastAPI(title="Spam Detector API", version="0.1")

class EmailIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    label: int
    spam_probability: float
    explain: str = None  # optional short reason

@app.get("/")
def root():
    return {"msg": "Spam Detector API is running. POST /predict with {'text': '...'}"}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: EmailIn):
    text = clean_text(payload.text)
    vec = vectorizer.transform([text])
    prob = float(model.predict_proba(vec)[0][1])
    label = 1 if prob >= THRESHOLD else 0

    # simple explain: top tokens contributing to spam (approx)
    try:
        # naive explanation: highest TF-IDF tokens in the input
        feature_names = vectorizer.get_feature_names_out()
        row = vec.tocoo()
        top_idx = sorted(zip(row.col, row.data), key=lambda x: x[1], reverse=True)[:5]
        top_tokens = [feature_names[i] for i,_ in top_idx]
        explain = f"Top tokens: {top_tokens}"
    except Exception:
        explain = None

    return {"label": int(label), "spam_probability": prob, "explain": explain}