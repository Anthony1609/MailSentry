from flask import Flask, render_template, request, jsonify
import pickle, os, re, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

app  = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE, "model.pkl")
VEC_PATH   = os.path.join(BASE, "vectorizer.pkl")
SUM_PATH   = os.path.join(BASE, "model_summary.pkl")

# ── Auto-train if model files missing ─────────────────────────
if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
    print("Model not found — training now...")
    try:
        from spam_detector import load_data, extract_features, train_and_evaluate, save_best
        df          = load_data()
        X, y, vec   = extract_features(df)
        results, *_ = train_and_evaluate(X, y)
        save_best(results, vec)
        print("Model trained and saved.")
    except Exception as e:
        print(f"Training error: {e}")

# ── Load model ────────────────────────────────────────────────
stemmer    = PorterStemmer()
stop_words = set(stopwords.words("english"))

try:
    model      = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VEC_PATH,   "rb"))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Could not load model: {e}")
    model      = None
    vectorizer = None


# ── Preprocessing ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+",             " httplink ",     text)
    text = re.sub(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", " emailaddr ",    text)
    text = re.sub(r"\$[\d,]+|£[\d,]+|€[\d,]+",  " currencyamt ",  text)
    text = re.sub(r"\b\d{10,}\b",                " phonenumber ",  text)
    text = re.sub(r"!{2,}",                      " multiexclaim ", text)
    text = re.sub(r"\?{2,}",                     " multiquestion ",text)
    text = re.sub(r"\d+",                        " numbertoken ",  text)
    text = re.sub(r"[^\w\s]",                    " ",              text)
    text = re.sub(r"\s+",                        " ",              text).strip()
    words = [stemmer.stem(w) for w in text.split()
             if w not in stop_words and len(w) > 1]
    return " ".join(words)


# ── Pure ML prediction — no artificial boosting ───────────────
def predict_email(text: str):
    if model is None or vectorizer is None:
        raise Exception("Model not loaded.")

    clean     = preprocess(text)
    feat      = vectorizer.transform([clean]).toarray()
    proba     = model.predict_proba(feat)[0]
    spam_prob = float(proba[1])
    pred      = 1 if spam_prob >= 0.40 else 0

    confidence = round((spam_prob if pred == 1 else proba[0]) * 100, 2)

    if pred == 0:
        risk = "SAFE"
    elif spam_prob < 0.55:
        risk = "LOW"
    elif spam_prob < 0.70:
        risk = "MEDIUM"
    elif spam_prob < 0.85:
        risk = "HIGH"
    else:
        risk = "CRITICAL"

    feat_names = vectorizer.get_feature_names_out()
    scores     = feat[0]
    top_idx    = scores.argsort()[::-1][:8]
    keywords   = [feat_names[i] for i in top_idx if scores[i] > 0]

    return {
        "label"     : "SPAM" if pred == 1 else "HAM",
        "risk"      : risk,
        "confidence": confidence,
        "spam_prob" : round(spam_prob * 100, 2),
        "keywords"  : keywords,
    }


# ── Model summary ─────────────────────────────────────────────
def get_summary():
    try:
        return pickle.load(open(SUM_PATH, "rb"))
    except:
        return {
            "Naive Bayes":            {"accuracy": 0.9767, "cv_mean": 0.9768, "cv_std": 0.0024},
            "Logistic Regression":    {"accuracy": 0.9731, "cv_mean": 0.9672, "cv_std": 0.0026},
            "Support Vector Machine": {"accuracy": 0.9821, "cv_mean": 0.9798, "cv_std": 0.0019},
        }


STATS = {
    "total": 25572, "ham": 14825, "spam": 10747,
    "features": 8000, "algorithms": 3, "threshold": 40,
}

DEVELOPER = {
    "name":    "Chukwuemeka Anthony Somtochukwu",
    "course":  "Computer Science",
    "school":  "Federal University of Technology Owerri (FUTO)",
    "email":   "chukwuemekaanthony73@gmail.com",
    "github":  "Anthony1609",
    "year":    "2025/2026",
    "project": "CSC 309 Mini Project #3",
    "bio":     "A passionate Computer Science student at FUTO with a keen interest in Artificial Intelligence, Machine Learning, and Software Development. This project demonstrates the application of supervised learning and natural language processing to solve a real-world problem.",
    "skills":  ["Python", "Machine Learning", "NLP", "Flask", "Scikit-learn", "NLTK", "HTML/CSS", "JavaScript"],
    "tools":   ["Naive Bayes", "Logistic Regression", "SVM", "TF-IDF Vectorizer", "NLTK Stemmer"],
}


# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detector")
def detector():
    return render_template("detector.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", stats=STATS, models=get_summary())

@app.route("/developer")
def developer():
    return render_template("developer.html", dev=DEVELOPER)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        return jsonify(predict_email(text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_bulk", methods=["POST"])
def predict_bulk():
    data  = request.get_json()
    texts = data.get("texts", [])
    if not texts:
        return jsonify({"error": "No texts provided"}), 400
    results = []
    for i, text in enumerate(texts[:20]):
        if text.strip():
            try:
                r = predict_email(text.strip())
                r["index"]   = i + 1
                r["preview"] = text.strip()[:80] + ("..." if len(text) > 80 else "")
                results.append(r)
            except:
                pass
    return jsonify({"results": results, "total": len(results)})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
