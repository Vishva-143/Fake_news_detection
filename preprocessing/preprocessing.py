"""Data loading and feature preparation for fake news classification."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from preprocessing.cleaning import clean_text

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"


def load_dataset(train_path: str | None = None, test_path: str | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and testing CSV files from disk."""
    train_file = Path(train_path) if train_path else DATASET_DIR / "train.csv"
    test_file = Path(test_path) if test_path else DATASET_DIR / "test.csv"

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    return train_df, test_df


def prepare_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[list[str], list[str], list[int], list[int]]:
    """Clean text data and return train/test features and labels."""
    for frame in (train_df, test_df):
        frame["text"] = frame["text"].fillna("")
        if "title" in frame.columns:
            frame["text"] = frame["title"].fillna("") + " " + frame["text"].fillna("")
        frame["clean_text"] = frame["text"].apply(clean_text)

    X_train = train_df["clean_text"].tolist()
    X_test = test_df["clean_text"].tolist()
    y_train = train_df["label"].astype(int).tolist()
    y_test = test_df["label"].astype(int).tolist()
    return X_train, X_test, y_train, y_test


def build_vectorizer(method: str = "tfidf", max_features: int = 5000):
    """Create a vectorizer for Bag-of-Words or TF-IDF features."""
    if method == "bow":
        return CountVectorizer(max_features=max_features, ngram_range=(1, 2))
    if method == "tfidf":
        return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    raise ValueError("method must be either 'bow' or 'tfidf'")


def transform_features(vectorizer, X_train: list[str], X_test: list[str]):
    """Transform text into numeric feature matrices."""
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    return X_train_features, X_test_features
