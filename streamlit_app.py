"""Streamlit demo UI for the phishing-detection model.
Run with: streamlit run streamlit_app.py
Loads the serialized model directly (no API dependency) for a simple, 
self-contained demo; point it at the FastAPI service instead (via `requests`)
if you want to demo the deployed API specifically.
"""
import json
import re
from pathlib import Path

import joblib
import streamlit as st

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

def clean_text(text):
    text = str(text).lower()
    text = _html_re.sub(" ", text)
    text = _url_re.sub(" <url> ", text)
    text = _email_re.sub(" <email> ", text)
    text = _num_re.sub(" <num> ", text)
    text = re.sub(r"[^a-z0-9<>!?$%. ]", " ", text)
    return _ws_re.sub(" ", text).strip()

st.title("Phishing Email Detector")
st.caption("TF-IDF + Logistic Regression reference model (see the research notebook "
           "for the full model comparison, calibration and threshold analysis).")

email_text = st.text_area("Paste an email body:", height=200)
if st.button("Score email") and email_text.strip():
    proba = float(model.predict_proba(vectorizer.transform([clean_text(email_text)]))[0, 1])
    label = "PHISHING" if proba >= THRESHOLD else "legit"
    st.metric("Phishing probability", f"{proba:.1%}")
    if label == "PHISHING":
        st.error(f"Flagged as {label} (threshold={THRESHOLD:.2f})")
    else:
        st.success(f"Classified as {label} (threshold={THRESHOLD:.2f})")
