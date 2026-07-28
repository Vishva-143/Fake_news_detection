"""Text cleaning utilities for news article preprocessing."""

import re
import string
from typing import List

import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    word_tokenize("sample")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    WordNetLemmatizer().lemmatize("tests")
except LookupError:
    nltk.download("wordnet", quiet=True)


STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()


def remove_urls(text: str) -> str:
    """Remove URLs from the input text."""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_html_tags(text: str) -> str:
    """Remove HTML tags and decode html entities."""
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def clean_text(text: str, use_stemming: bool = False) -> str:
    """Clean and normalize text for machine learning."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in STOP_WORDS and len(token) > 1]

    if use_stemming:
        tokens = [STEMMER.stem(token) for token in tokens]
    else:
        tokens = [LEMMATIZER.lemmatize(token) for token in tokens]

    return " ".join(tokens)
