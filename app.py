from flask import Flask, render_template, request, jsonify
import pickle, os, re, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from spam_detector import predict_email

nltk.download("stopwords", quiet=True)
app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Load model summary if available ──────────────────────────
def get_summary():
    try:
        return pickle.load(open(os.path.join(BASE,"model_summary.pkl"),"rb"))
    except:
        return {
            "Naive Bayes":         {"accuracy":0.9767,"cv_mean":0.9768,"cv_std":0.0024},
            "Logistic Regression": {"accuracy":0.9731,"cv_mean":0.9672,"cv_std":0.0026},
            "Support Vector Machine": {"accuracy":0.9821,"cv_mean":0.9798,"cv_std":0.0019},
        }

STATS = {
    "total": 5572, "ham": 4825, "spam": 747,
    "features": 5000, "algorithms": 3, "threshold": 30,
}

DEVELOPER = {
    "name":   "Chukwuemeka Anthony Somtochukwu",
    "course": "Computer Science",
    "school": "Federal University of Technology Owerri (FUTO)",
    "email":  "chukwuemekaanthony73@gmail.com",
    "github": "Anthony1609",
    "year":   "2025/2026",
    "project": "CSC 309 Mini Project #3",
    "bio": "A passionate Computer Science student at FUTO with a keen interest in Artificial Intelligence, Machine Learning, and Software Development. This project demonstrates the application of supervised learning and natural language processing to solve a real-world problem.",
    "skills": ["Python", "Machine Learning", "NLP", "Flask", "Scikit-learn", "NLTK", "HTML/CSS", "JavaScript"],
    "tools":  ["Naive Bayes", "Logistic Regression", "SVM", "TF-IDF Vectorizer", "NLTK Stemmer"],
}

# ── ROUTES ───────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detector")
def detector():
    return render_template("detector.html")

@app.route("/dashboard")
def dashboard():
    summary = get_summary()
    return render_template("dashboard.html", stats=STATS, models=summary)

@app.route("/developer")
def developer():
    return render_template("developer.html", dev=DEVELOPER)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text","").strip()
    if not text:
        return jsonify({"error":"No text provided"}), 400
    try:
        result = predict_email(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_bulk", methods=["POST"])
def predict_bulk():
    data  = request.get_json()
    texts = data.get("texts", [])
    if not texts:
        return jsonify({"error":"No texts provided"}), 400
    results = []
    for i, text in enumerate(texts[:20]):  # max 20
        if text.strip():
            try:
                r = predict_email(text.strip())
                r["index"] = i + 1
                r["preview"] = text.strip()[:80] + ("..." if len(text)>80 else "")
                results.append(r)
            except:
                pass
    return jsonify({"results": results, "total": len(results)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
