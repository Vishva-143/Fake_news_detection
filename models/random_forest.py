"""Random Forest model wrapper."""

from sklearn.ensemble import RandomForestClassifier


def build_random_forest() -> RandomForestClassifier:
    """Create a random forest classifier."""
    return RandomForestClassifier(n_estimators=200, random_state=42)
