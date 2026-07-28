"""Neural network model wrapper."""

from sklearn.neural_network import MLPClassifier


def build_neural_network() -> MLPClassifier:
    """Create a multilayer perceptron classifier."""
    return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
