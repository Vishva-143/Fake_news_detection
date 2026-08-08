"""Flask web application for fake news detection."""

from flask import Flask, render_template, request

from predict import predict_news
from utils.dashboard import get_dashboard_payload

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the home page and prediction result."""
    result = None
    article = ""
    if request.method == "POST":
        article = request.form.get("article", "")
        if article.strip():
            try:
                label, confidence, model_name = predict_news(article)
                result = {
                    "prediction": label,
                    "confidence": confidence,
                    "model_used": model_name,
                    "article": article,
                }
            except Exception as exc:  # pragma: no cover - defensive error handling
                result = {"error": f"Prediction failed: {exc}"}
    return render_template("index.html", result=result, article=article)


@app.route("/analytics")
def analytics():
    """Render the analytics dashboard with dynamic charts and summary cards."""
    payload = get_dashboard_payload()
    return render_template("analytics.html", **payload)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)