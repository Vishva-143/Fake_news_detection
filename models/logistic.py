"""Logistic Regression model wrapper."""

from sklearn.linear_model import LogisticRegression


def build_logistic_regression() -> LogisticRegression:
    """Create a logistic regression classifier."""
    return LogisticRegression(max_iter=2000, random_state=42)
