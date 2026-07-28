"""Train and evaluate multiple fake news classifiers."""

import pickle
from pathlib import Path

import pandas as pd

from models.knn import build_knn
from models.logistic import build_logistic_regression
from models.neural_network import build_neural_network
from models.random_forest import build_random_forest
from preprocessing.preprocessing import build_vectorizer, load_dataset, prepare_datasets, transform_features
from utils.metrics import build_comparison_table, evaluate_model

BASE_DIR = Path(__file__).resolve().parent
SAVED_MODELS_DIR = BASE_DIR / "saved_models"


def train_models() -> dict:
    """Train logistic regression, KNN, random forest, and MLP models."""
    train_df, test_df = load_dataset()
    X_train, X_test, y_train, y_test = prepare_datasets(train_df, test_df)

    vectorizer = build_vectorizer(method="tfidf")
    X_train_features, X_test_features = transform_features(vectorizer, X_train, X_test)

    models = {
        "Logistic Regression": build_logistic_regression(),
        "KNN": build_knn(),
        "Random Forest": build_random_forest(),
        "Neural Network": build_neural_network(),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_features, y_train)
        metrics = evaluate_model(model, X_test_features, y_test)
        results[name] = metrics
        trained_models[name] = model

    comparison_table = build_comparison_table(results)
    comparison_table.to_csv(BASE_DIR / "metrics.csv", index=True)
    print("\n=== Model Comparison ===")
    print(comparison_table)

    best_model_name = comparison_table["accuracy"].idxmax()
    best_model = trained_models[best_model_name]
    print(f"\n✓ Best model: {best_model_name}")

    with open(SAVED_MODELS_DIR / "logistic.pkl", "wb") as handle:
        pickle.dump(best_model, handle)
    with open(SAVED_MODELS_DIR / "vectorizer.pkl", "wb") as handle:
        pickle.dump(vectorizer, handle)

    print(f"✓ Models saved to {SAVED_MODELS_DIR}")
    return {"results": results, "best_model": best_model_name}


if __name__ == "__main__":
    train_models()
