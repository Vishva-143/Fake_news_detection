"""Prediction utilities for the Flask web app."""

import pickle
from pathlib import Path

from preprocessing.cleaning import clean_text

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_models" / "logistic.pkl"
VECTORIZER_PATH = BASE_DIR / "saved_models" / "vectorizer.pkl"


def load_model_artifacts():
    """Load the trained model and vectorizer from disk."""
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    with open(VECTORIZER_PATH, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer


def predict_news(article: str):
    """Predict whether an article is fake or real."""
    model, vectorizer = load_model_artifacts()
    cleaned_text = clean_text(article)
    features = vectorizer.transform([cleaned_text])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].max()
    label = "Real News" if prediction == 1 else "Fake News"
    return label, round(float(probability), 4), type(model).__name__
