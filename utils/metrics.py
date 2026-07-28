"""Evaluation helpers and reporting utilities."""

from typing import Dict, List

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_model(model, X_test, y_test, label_names: List[str] | None = None) -> Dict[str, float]:
    """Return evaluation metrics for a trained classifier."""
    y_pred = model.predict(X_test)
    label_names = label_names or ["Fake", "Real"]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def plot_confusion_matrix(y_true, y_pred, labels: List[str] | None = None) -> None:
    """Plot a confusion matrix heatmap."""
    pass


def plot_roc_curve(model, X_test, y_test) -> None:
    """Plot ROC curve for binary classification."""
    pass



def build_comparison_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Convert evaluation results into a comparison table."""
    return pd.DataFrame(results).T


def generate_classification_report(y_true, y_pred) -> str:
    """Return text classification report."""
    return classification_report(y_true, y_pred)
