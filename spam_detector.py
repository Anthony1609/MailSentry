# ============================================================
#  MailSentry AI — Improved Spam Detection Engine
#  CSC 309 Mini Project #3
#  Algorithms : Naive Bayes, Logistic Regression, LinearSVC
#  Tools      : Python, Scikit-learn, NLTK
# ============================================================

import os, re, pickle, urllib.request
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

DATASET_URL  = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
DATASET_PATH = "emails.csv"

stemmer    = PorterStemmer()
stop_words = set(stopwords.words("english"))

# ── Known spam trigger words (boosts detection for long emails) ──
SPAM_TRIGGERS = [
    "free", "win", "winner", "won", "prize", "claim", "click", "urgent",
    "congratulations", "selected", "offer", "limited", "act now", "call now",
    "order now", "buy now", "risk free", "guaranteed", "no obligation",
    "credit card", "cash", "earn", "income", "profit", "investment",
    "million", "billion", "lottery", "casino", "viagra", "cialis", "pharmacy",
    "prescription", "medication", "weight loss", "diet", "enlarge",
    "verify", "suspended", "account", "banned", "confirm identity",
    "click here", "unsubscribe", "remove", "dear customer", "dear user",
    "final notice", "last chance", "expire", "warning", "alert",
    "bank", "paypal", "apple", "microsoft", "google", "amazon",
    "password", "login", "sign in", "update required",
    "make money", "work from home", "business opportunity", "mlm",
    "referral", "commission", "downline", "matrix", "safelist",
]


# ── LOAD DATA ────────────────────────────────────────────────
def load_data():
    if not os.path.exists(DATASET_PATH):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATASET_URL, "sms_raw.tsv")
        df = pd.read_csv("sms_raw.tsv", sep="\t", header=None,
                         names=["label", "message"])
        df.to_csv(DATASET_PATH, index=False)
    else:
        df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded — {len(df)} rows")
    print(df["label"].value_counts())
    return df


# ── PREPROCESSING ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " urllink urllink urllink ", text)  # URLs = strong spam signal
    text = re.sub(r"\$[\d,]+|\d+[\$£€]", " moneysign moneysign ", text)  # money amounts
    text = re.sub(r"!{2,}", " exclamation exclamation ", text)           # multiple !!!
    text = re.sub(r"\d+", " numtoken ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [stemmer.stem(w) for w in text.split()
             if w not in stop_words and len(w) > 1]
    return " ".join(words)


def count_spam_triggers(text: str) -> int:
    """Count how many known spam trigger phrases appear in the text."""
    text_lower = text.lower()
    return sum(1 for t in SPAM_TRIGGERS if t in text_lower)


# ── FEATURE EXTRACTION ────────────────────────────────────────
def extract_features(df):
    print("Preprocessing text...")
    df["clean"] = df["message"].apply(preprocess)
    vec = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X = vec.fit_transform(df["clean"]).toarray()
    y = df["label"].map({"spam": 1, "ham": 0}).values
    print(f"Feature matrix: {X.shape}")
    return X, y, vec


# ── TRAIN ALL 3 MODELS ────────────────────────────────────────
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    svc_base = LinearSVC(max_iter=2000, C=1.0)
    models = {
        "Naive Bayes":            MultinomialNB(),
        "Logistic Regression":    LogisticRegression(max_iter=1000, C=1.0),
        "Support Vector Machine": CalibratedClassifierCV(svc_base),
    }

    results = {}
    for name, model in models.items():
        print(f"\n{'='*50}\n  {name}\n{'='*50}")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        cv    = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        print(f"  Test Acc : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"  CV  Acc  : {cv.mean():.4f} ± {cv.std():.4f}")
        print(classification_report(y_test, preds, target_names=["Ham", "Spam"]))
        results[name] = {
            "model": model, "accuracy": acc,
            "cv_mean": cv.mean(), "cv_std": cv.std(),
            "preds": preds, "y_test": y_test,
        }
    return results, X_test, y_test


# ── SAVE BEST MODEL ───────────────────────────────────────────
def save_best(results, vectorizer):
    best = max(results, key=lambda n: results[n]["accuracy"])
    pickle.dump(results[best]["model"], open("model.pkl", "wb"))
    pickle.dump(vectorizer,             open("vectorizer.pkl", "wb"))
    summary = {
        name: {
            "accuracy": r["accuracy"],
            "cv_mean":  r["cv_mean"],
            "cv_std":   r["cv_std"],
        }
        for name, r in results.items()
    }
    pickle.dump(summary, open("model_summary.pkl", "wb"))
    print(f"\nBest model saved: {best}")
    return best


# ── PREDICT ───────────────────────────────────────────────────
def predict_email(text: str):
    model      = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

    clean     = preprocess(text)
    feat      = vectorizer.transform([clean]).toarray()
    proba     = model.predict_proba(feat)[0]
    spam_prob = float(proba[1])

    # ── Boost spam probability based on trigger words ──────
    trigger_count = count_spam_triggers(text)
    if trigger_count >= 5:
        spam_prob = min(1.0, spam_prob + 0.35)
    elif trigger_count >= 3:
        spam_prob = min(1.0, spam_prob + 0.20)
    elif trigger_count >= 1:
        spam_prob = min(1.0, spam_prob + 0.10)

    # ── URL count boost ────────────────────────────────────
    url_count = len(re.findall(r"http\S+|www\S+", text.lower()))
    if url_count >= 3:
        spam_prob = min(1.0, spam_prob + 0.15)
    elif url_count >= 1:
        spam_prob = min(1.0, spam_prob + 0.05)

    # ── Exclamation marks boost ────────────────────────────
    exclaim = text.count("!")
    if exclaim >= 5:
        spam_prob = min(1.0, spam_prob + 0.10)

    # ── ALL CAPS ratio boost ───────────────────────────────
    words      = text.split()
    caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1)
    if caps_ratio > 0.3:
        spam_prob = min(1.0, spam_prob + 0.15)

    # ── Final classification with lowered threshold ────────
    pred       = 1 if spam_prob >= 0.25 else 0
    confidence = round((spam_prob if pred == 1 else (1 - spam_prob)) * 100, 2)

    # ── Risk level ─────────────────────────────────────────
    if pred == 0:
        risk = "SAFE"
    elif spam_prob < 0.45:
        risk = "LOW"
    elif spam_prob < 0.65:
        risk = "MEDIUM"
    elif spam_prob < 0.85:
        risk = "HIGH"
    else:
        risk = "CRITICAL"

    # ── Top keywords ───────────────────────────────────────
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


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    df           = load_data()
    X, y, vec    = extract_features(df)
    results, *_  = train_and_evaluate(X, y)
    best         = save_best(results, vec)
    print(f"\nBest model: {best}")

    tests = [
        "CONGRATULATIONS! You have won $1,000,000. Click here NOW to claim!!!",
        "Hey are we still meeting tomorrow for the project?",
        "URGENT: Your account has been suspended. Verify now or lose access!",
        "Please find the attached Q3 report for your review.",
        "FREE Viagra pills! No prescription needed! Order now! Limited offer!",
        "Don't forget to submit your CSC 309 project by Friday.",
        "Join my team today and earn from referrals. MLM commissions on multiple levels. Click to earn credits.",
        "Hi, just checking if you received my last email about the meeting.",
    ]
    print("\n" + "="*55 + "\nDEMO PREDICTIONS\n" + "="*55)
    for t in tests:
        r = predict_email(t)
        print(f"\n{t[:65]}")
        print(f"  → {r['label']} | Risk: {r['risk']} | Confidence: {r['confidence']}% | Triggers: {count_spam_triggers(t)}")
