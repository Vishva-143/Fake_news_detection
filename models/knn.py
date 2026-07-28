"""K-Nearest Neighbors model wrapper."""

from sklearn.neighbors import KNeighborsClassifier


def build_knn() -> KNeighborsClassifier:
    """Create a KNN classifier."""
    return KNeighborsClassifier(n_neighbors=5)
