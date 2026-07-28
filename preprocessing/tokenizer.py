"""Tokenization helpers used by the preprocessing pipeline."""

from nltk.tokenize import word_tokenize


def tokenize_text(text: str) -> list[str]:
    """Split text into lowercase tokens."""
    if not isinstance(text, str):
        return []
    return word_tokenize(text.lower())
