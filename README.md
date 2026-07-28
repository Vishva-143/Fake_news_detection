<<<<<<< HEAD
# Fake_news_detection
=======
# AI-Powered Fake News Detection Using Text Classification

## Introduction
This project builds a beginner-friendly machine learning system that detects whether a news article is fake or real using text preprocessing and classification models.

## Problem Statement
The spread of fake news has become a major concern in modern society. This project aims to automate the classification of news articles using natural language processing and machine learning.

## Objectives
- Preprocess raw text data effectively
- Train multiple classification models
- Compare model performance using evaluation metrics
- Deploy a simple Flask-based web application for predictions

## Features
- Text cleaning and normalization
- TF-IDF feature extraction
- Training of multiple classifiers
- Model comparison and visualization
- Flask web interface for real-time predictions

## Technology Stack
- Python 3.12
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Flask
- HTML/CSS/Bootstrap 5/JavaScript

## Dataset
The project is designed for the Kaggle Fake News Detection dataset. Place the files in the dataset folder as:
- dataset/train.csv
- dataset/test.csv

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run
```bash
python train.py
python app.py
```
Then open http://127.0.0.1:5000/

## Project Structure
```text
Fake-News-Detection/
├── dataset/
├── notebooks/
├── preprocessing/
├── models/
├── utils/
├── saved_models/
├── static/
├── templates/
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── LICENSE
```

## Screenshots
Add screenshots of the web interface and results here.

## Future Scope
- Deploy on cloud services
- Add more advanced transformer-based models
- Build API endpoints for integration

## Contributors
- Your Name

## License
This project is licensed under the MIT License.
>>>>>>> ef8f3ce (Commit all changes before code migration completed: 2026-07-28 11:42:25)
