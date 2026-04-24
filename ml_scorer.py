"""
ML-based comment scoring. Loads trained model, scores new comments.
Also runs rule-based litigation risk detection.
"""
import joblib
import re
import os
import numpy as np

MODEL_PATH = "model/significance_clf.pkl"
_model = None


def get_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
        else:
            raise FileNotFoundError("Model not found. Run: python train.py first.")
    return _model


def score_significance(text: str) -> dict:
    """
    ML significance score using trained TF-IDF + LogisticRegression.
    Returns score 0-100, significant bool, and confidence.
    """
    model = get_model()
    prob = model.predict_proba([text])[0]
    sig_prob = float(prob[1])
    score = round(sig_prob * 100)
    return {
        "score": score,
        "significant": sig_prob >= 0.5,
        "confidence": "high" if abs(sig_prob - 0.5) > 0.3 else "medium",
    }


# Litigation risk patterns drawn from real APA case law
LIT_PATTERNS = [
    (r"\d+ U\.S\. \d+",                     "Case citation"),
    (r"\d+ U\.S\.C\. §\d+",                "Statute citation"),
    (r"5 U\.S\.C\. §706",                   "APA §706 — judicial review"),
    (r"reserve all rights",                  "Explicit legal threat"),
    (r"seek judicial review",                "Explicit legal threat"),
    (r"arbitrary and capricious",            "A&C challenge"),
    (r"major questions doctrine",            "Major questions doctrine"),
    (r"(Fifth|D\.C\.) Circuit",             "Forum selection signal"),
    (r"injunctive relief",                   "Injunction threat"),
    (r"(LLP|Law Firm|Counsel|Attorney)",     "Law firm author"),
    (r"Executive Order 12866",               "EO 12866 — cost-benefit"),
    (r"Regulatory Flexibility Act|RFA",      "RFA challenge"),
]


def score_litigation(text: str, author: str = "") -> dict:
    """
    Rule-based litigation risk scorer.
    Returns score 0-100, risk level, and list of triggered flags.
    """
    combined = text + " " + author
    triggered = []
    for pattern, label in LIT_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            triggered.append(label)

    score = min(100, round(len(triggered) / len(LIT_PATTERNS) * 100))
    risk = "high" if score >= 50 else "medium" if score >= 25 else "low"
    return {
        "score": score,
        "risk": risk,
        "flags": triggered,
    }


def detect_form_letter(comments: list) -> list:
    """
    Detect coordinated form letter campaigns by text similarity.
    Groups near-identical comments (same first 80 chars) together.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for c in comments:
        key = re.sub(r"\s+", " ", c["text"][:80].lower().strip())
        groups[key].append(c["id"])

    campaign_map = {}
    for key, ids in groups.items():
        if len(ids) > 1:
            for cid in ids:
                campaign_map[cid] = {"campaign_size": len(ids), "is_form_letter": True}

    result = []
    for c in comments:
        c["form_letter"] = campaign_map.get(c["id"], {"is_form_letter": False})
        result.append(c)
    return result


def extract_cfr_sections(text: str) -> list:
    """Extract CFR section references from comment text."""
    pattern = r"§\s*[\d]+\.[\d]+[\w\-]*"
    matches = re.findall(pattern, text)
    return list(set(matches)) if matches else ["General"]


def score_comment(comment: dict) -> dict:
    """Run full scoring pipeline on a single comment."""
    text = comment.get("text", "")
    author = comment.get("name", "")
    sig = score_significance(text)
    lit = score_litigation(text, author)
    cfr = extract_cfr_sections(text)
    return {
        **comment,
        "significance_score": sig["score"],
        "significant": sig["significant"],
        "sig_confidence": sig["confidence"],
        "litigation_score": lit["score"],
        "litigation_risk": lit["risk"],
        "litigation_flags": lit["flags"],
        "cfr_sections": cfr,
    }
