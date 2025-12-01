import joblib
import re

# Load model + vectorizer
model = joblib.load("models/spam_model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")

THRESHOLD = 0.5  # use the threshold you selected

def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"\r|\n", " ", s)
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"http\S+|www\.\S+", " URL ", s)
    s = re.sub(r"\S+@\S+", " EMAIL ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Ask user for input
email_text = input("Paste your email text here:\n\n")

# Clean → vectorize → predict
cleaned = clean_text(email_text)
vec = vectorizer.transform([cleaned])
prob = model.predict_proba(vec)[0][1]
label = 1 if prob >= THRESHOLD else 0

print("\n================ RESULT ================")
print("SPAM Probability:", prob)
print("Prediction:", "SPAM ❌" if label == 1 else "NOT SPAM ✅")
print("========================================")