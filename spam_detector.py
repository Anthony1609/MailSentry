# ============================================================
#  MailSentry AI — Professional Spam Detection Engine
#  CSC 309 Mini Project #3
#  Dataset  : Enron Email Dataset (33,000+ real emails)
#             + SMS Spam Collection (5,572 messages)
#  Algorithms: Naive Bayes, Logistic Regression, LinearSVC
#  Tools    : Python, Scikit-learn, NLTK
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

# ── Dataset URLs ──────────────────────────────────────────────
SMS_URL     = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
EMAIL_URL   = "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/main/enron_spam_data.csv"
SMS_PATH    = "sms_raw.tsv"
EMAIL_PATH  = "enron_emails.csv"
COMBINED    = "emails.csv"

stemmer    = PorterStemmer()
stop_words = set(stopwords.words("english"))


# ── LOAD & COMBINE DATASETS ───────────────────────────────────
def load_data():
    frames = []

    # 1. SMS Spam Collection
    if not os.path.exists(SMS_PATH):
        print("Downloading SMS Spam Collection...")
        urllib.request.urlretrieve(SMS_URL, SMS_PATH)
    sms_df = pd.read_csv(SMS_PATH, sep="\t", header=None, names=["label","message"])
    sms_df = sms_df[["label","message"]]
    frames.append(sms_df)
    print(f"SMS dataset: {len(sms_df)} rows")

    # 2. Enron Email Dataset
    if not os.path.exists(EMAIL_PATH):
        print("Downloading Enron Email Dataset (this may take a moment)...")
        try:
            urllib.request.urlretrieve(EMAIL_URL, EMAIL_PATH)
            print("Enron dataset downloaded.")
        except Exception as e:
            print(f"Could not download Enron dataset: {e}")

    if os.path.exists(EMAIL_PATH):
        try:
            email_df = pd.read_csv(EMAIL_PATH)
            # Enron dataset columns: Message ID, Subject, Message, Spam/Ham
            if "Message" in email_df.columns and "Spam/Ham" in email_df.columns:
                email_df = email_df[["Spam/Ham", "Message"]].copy()
                email_df.columns = ["label", "message"]
                # Combine subject + message if Subject exists
            elif "Subject" in email_df.columns and "Message" in email_df.columns:
                email_df["message"] = email_df["Subject"].fillna("") + " " + email_df["Message"].fillna("")
                email_df["label"]   = email_df["Spam/Ham"] if "Spam/Ham" in email_df.columns else email_df["label"]
                email_df = email_df[["label","message"]]

            # Normalize labels
            email_df["label"] = email_df["label"].str.strip().str.lower()
            email_df = email_df[email_df["label"].isin(["spam","ham"])]
            email_df = email_df.dropna(subset=["message"])
            email_df["message"] = email_df["message"].astype(str)

            # Balance — use up to 10,000 of each class
            spam_df = email_df[email_df["label"]=="spam"].sample(
                min(10000, len(email_df[email_df["label"]=="spam"])), random_state=42)
            ham_df  = email_df[email_df["label"]=="ham"].sample(
                min(10000, len(email_df[email_df["label"]=="ham"])), random_state=42)
            email_balanced = pd.concat([spam_df, ham_df])
            frames.append(email_balanced)
            print(f"Enron dataset: {len(email_balanced)} rows")
        except Exception as e:
            print(f"Could not process Enron dataset: {e}")

    # 3. Combine all
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["message","label"])
    df["message"] = df["message"].astype(str)
    df["label"]   = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["spam","ham"])]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    df.to_csv(COMBINED, index=False)
    print(f"\nCombined dataset: {len(df)} total rows")
    print(df["label"].value_counts())
    return df


# ── PREPROCESSING ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = str(text).lower()
    # Preserve spam signal patterns before removing
    text = re.sub(r"http\S+|www\S+", " httplink ", text)
    text = re.sub(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", " emailaddr ", text)
    text = re.sub(r"\$[\d,]+|£[\d,]+|€[\d,]+", " currencyamt ", text)
    text = re.sub(r"\b\d{10,}\b", " phonenumber ", text)
    text = re.sub(r"!{2,}", " multiexclaim ", text)
    text = re.sub(r"\?{2,}", " multiquestion ", text)
    text = re.sub(r"\d+", " numbertoken ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [stemmer.stem(w) for w in text.split()
             if w not in stop_words and len(w) > 1]
    return " ".join(words)


# ── FEATURE EXTRACTION ────────────────────────────────────────
def extract_features(df):
    print("\nPreprocessing text...")
    df["clean"] = df["message"].apply(preprocess)
    vec = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
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

    svc_base = LinearSVC(max_iter=3000, C=1.0)
    models = {
        "Naive Bayes":            MultinomialNB(alpha=0.1),
        "Logistic Regression":    LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
        "Support Vector Machine": CalibratedClassifierCV(svc_base),
    }

    results = {}
    for name, mdl in models.items():
        print(f"\n{'='*52}\n  {name}\n{'='*52}")
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        cv    = cross_val_score(mdl, X, y, cv=5, scoring="accuracy")
        print(f"  Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"  CV  Accuracy  : {cv.mean():.4f} ± {cv.std():.4f}")
        print(classification_report(y_test, preds, target_names=["Ham","Spam"]))
        results[name] = {
            "model":   mdl,
            "accuracy": acc,
            "cv_mean": cv.mean(),
            "cv_std":  cv.std(),
            "preds":   preds,
            "y_test":  y_test,
        }
    return results, X_test, y_test


# ── SAVE BEST MODEL ───────────────────────────────────────────
def save_best(results, vectorizer):
    best = max(results, key=lambda n: results[n]["accuracy"])
    pickle.dump(results[best]["model"], open("model.pkl",         "wb"))
    pickle.dump(vectorizer,             open("vectorizer.pkl",    "wb"))
    summary = {
        name: {
            "accuracy": r["accuracy"],
            "cv_mean":  r["cv_mean"],
            "cv_std":   r["cv_std"],
        }
        for name, r in results.items()
    }
    pickle.dump(summary, open("model_summary.pkl", "wb"))
    print(f"\n✅ Best model saved: {best}")
    return best


# ── PREDICT ───────────────────────────────────────────────────
def predict_email(text: str):
    mdl        = pickle.load(open("model.pkl",      "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

    clean     = preprocess(text)
    feat      = vectorizer.transform([clean]).toarray()
    proba     = mdl.predict_proba(feat)[0]
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


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    df          = load_data()
    X, y, vec   = extract_features(df)
    results, *_ = train_and_evaluate(X, y)
    best        = save_best(results, vec)

    print(f"\n{'='*52}\nDEMO PREDICTIONS\n{'='*52}")
    tests = [
        ("SPAM", "CONGRATULATIONS! You have won $1,000,000. Click here NOW to claim your prize!!!"),
        ("HAM",  "Hey, are we still meeting tomorrow for the project discussion?"),
        ("SPAM", "URGENT: Your account has been suspended. Verify immediately or lose access!"),
        ("HAM",  "Please find the attached Q3 progress report for your review."),
        ("SPAM", "Get FREE Viagra, Cialis and other medications. No prescription needed! Order now!"),
        ("HAM",  "Don't forget to submit your CSC 309 mini project by Friday."),
        ("SPAM", "Join my team today! Earn from referrals. MLM commissions on multiple levels."),
        ("HAM",  "Hi, just checking if you received the lecture notes I sent yesterday."),
        ("SPAM", "Dear customer, your bank account requires immediate verification. Click the link below."),
        ("SPAM", "Make money from home! Work just 2 hours a day and earn $500 daily. Limited spots!"),
    ]
    for expected, t in tests:
        r = predict_email(t)
        status = "✅" if r["label"] == expected else "❌"
        print(f"\n{status} [{expected}→{r['label']}] Risk:{r['risk']} Conf:{r['confidence']}%")
        print(f"   {t[:70]}")
