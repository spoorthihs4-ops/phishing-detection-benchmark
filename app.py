"""Phishing-detection scoring API. Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000
Expects tfidf_vectorizer.joblib, logreg_model.joblib and deploy_config.json
in the same directory (produced by Section 24.1 of the training notebook).
"""
import json
import re
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

HERE = Path(__file__).parent
vectorizer = joblib.load(HERE / "tfidf_vectorizer.joblib")
model = joblib.load(HERE / "logreg_model.joblib")
CONFIG = json.loads((HERE / "deploy_config.json").read_text())
THRESHOLD = CONFIG.get("threshold", 0.5)

_html_re = re.compile(r"<[^>]+>")
_url_re = re.compile(r"(https?://\S+|www\.\S+)")
_email_re = re.compile(r"\S+@\S+\.\S+")
_num_re = re.compile(r"\b\d{2,}\b")
_ws_re = re.compile(r"\s+")

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = _html_re.sub(" ", text)
    text = _url_re.sub(" <url> ", text)
    text = _email_re.sub(" <email> ", text)
    text = _num_re.sub(" <num> ", text)
    text = re.sub(r"[^a-z0-9<>!?$%. ]", " ", text)
    text = _ws_re.sub(" ", text).strip()
    return text

app = FastAPI(title="Phishing Detection API", version="1.0")

class EmailRequest(BaseModel):
    text: str

class EmailResponse(BaseModel):
    probability: float
    label: str
    threshold: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=EmailResponse)
def predict(req: EmailRequest):
    cleaned = clean_text(req.text)
    proba = float(model.predict_proba(vectorizer.transform([cleaned]))[0, 1])
    label = "phishing" if proba >= THRESHOLD else "legit"
    return EmailResponse(probability=proba, label=label, threshold=THRESHOLD)
