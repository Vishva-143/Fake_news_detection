"""Flask web application for fake news detection."""

from flask import Flask, render_template, request

from predict import predict_news

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the home page and prediction result."""
    result = None
    if request.method == "POST":
        article = request.form.get("article", "")
        if article.strip():
            try:
                label, confidence, model_name = predict_news(article)
                result = {
                    "prediction": label,
                    "confidence": confidence,
                    "model_used": model_name,
                }
            except Exception as exc:  # pragma: no cover - defensive error handling
                result = {"error": f"Prediction failed: {exc}"}
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
