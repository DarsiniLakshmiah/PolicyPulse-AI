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


# Organization type signals for author fingerprinting
_ORG_SIGNALS = [
    (r"\bLLP\b|\bLLC\b|\bP\.C\.\b|\bLaw (Firm|Group|Office|Partners)\b", "law_firm"),
    (r"\b(Alliance|Association|Coalition|Federation|Society|Institute|Foundation|Council)\b", "advocacy_org"),
    (r"\b(Chamber of Commerce|Business Roundtable|Industry (Association|Council|Group))\b", "industry_group"),
    (r"\b(Union|Workers?|AFL-?CIO|Labor)\b", "labor_org"),
]

_CITE_PATTERN = re.compile(
    r"\d+ U\.S\. \d+|\d+ U\.S\.C\.?\s*§\s*\d+|5 U\.S\.C\. §706|Executive Order \d+",
    re.IGNORECASE,
)


def _classify_org(name: str):
    for pattern, org_type in _ORG_SIGNALS:
        if re.search(pattern, name, re.IGNORECASE):
            return org_type
    return None


def _clean_org_name(name: str) -> str:
    """Extract the org portion from a compound name like 'Person — Org' or 'Person, Role'."""
    if " — " in name:
        parts = name.split(" — ")
        for p in reversed(parts):
            if re.search(r"\b(LLP|LLC|Alliance|Chamber|Roundtable|Association|Coalition|Foundation|Institute|Council|Society|Union)\b", p, re.IGNORECASE):
                return p.strip()
        return parts[-1].strip()
    if re.search(r"\b(LLP|LLC|Alliance|Chamber|Association|Coalition|Federation|Foundation|Institute|Council|Society)\b", name, re.IGNORECASE):
        return name
    return name


def build_comment_network(scored_comments: list) -> dict:
    """
    Build a comment coordination network graph.
    Detects: law firm authorship, form letter campaigns, shared legal citations.
    Returns {nodes, edges, stats} for D3 force-directed rendering.
    """
    from collections import defaultdict

    nodes: list = []
    edges: list = []
    org_registry: dict = {}      # org_name → node_id
    campaign_registry: dict = {} # fingerprint → node_id
    cite_registry: set = set()

    # ── Comment nodes ────────────────────────────────────
    for c in scored_comments:
        nodes.append({
            "id": c["id"],
            "type": "comment",
            "group": "comment",
            "name": c.get("name", "Anonymous"),
            "text_preview": c.get("text", "")[:100],
            "significance_score": c.get("significance_score", 0),
            "litigation_risk": c.get("litigation_risk", "low"),
            "significant": c.get("significant", False),
        })

    # ── Org authorship edges ──────────────────────────────
    for c in scored_comments:
        name = c.get("name", "")
        org_type = _classify_org(name)
        # Also check each part of compound names like "Person — Org"
        if not org_type and " — " in name:
            for part in name.split(" — "):
                org_type = _classify_org(part.strip())
                if org_type:
                    break

        if org_type:
            org_name = _clean_org_name(name)
            if org_name not in org_registry:
                org_id = f"org-{len(org_registry)}"
                org_registry[org_name] = org_id
                nodes.append({
                    "id": org_id,
                    "type": org_type,
                    "group": "orchestrator",
                    "name": org_name,
                    "comment_count": 0,
                })
            org_id = org_registry[org_name]
            for n in nodes:
                if n["id"] == org_id:
                    n["comment_count"] = n.get("comment_count", 0) + 1
                    break
            edges.append({"source": c["id"], "target": org_id, "type": "authored_by", "weight": 2})

    # ── Form letter campaign edges ────────────────────────
    campaign_groups: dict = defaultdict(list)
    for c in scored_comments:
        if c.get("form_letter", {}).get("is_form_letter"):
            key = re.sub(r"\s+", " ", c.get("text", "")[:80].lower().strip())
            campaign_groups[key].append(c)

    for key, group in campaign_groups.items():
        if len(group) >= 2:
            fp = str(abs(hash(key)) % 100000)
            if fp not in campaign_registry:
                camp_id = f"campaign-{fp}"
                campaign_registry[fp] = camp_id
                nodes.append({
                    "id": camp_id,
                    "type": "campaign",
                    "group": "orchestrator",
                    "name": f"Form Letter Campaign",
                    "comment_count": len(group),
                    "text_preview": group[0].get("text", "")[:80],
                })
            camp_id = campaign_registry[fp]
            for c in group:
                edges.append({"source": c["id"], "target": camp_id, "type": "form_letter", "weight": 1})

    # ── Shared citation edges ─────────────────────────────
    citation_map: dict = defaultdict(list)
    for c in scored_comments:
        for cite in set(_CITE_PATTERN.findall(c.get("text", ""))):
            citation_map[cite.strip()].append(c["id"])

    for cite, cids in citation_map.items():
        if len(cids) >= 2:
            cite_id = f"cite-{abs(hash(cite)) % 100000}"
            if cite_id not in cite_registry:
                cite_registry.add(cite_id)
                nodes.append({
                    "id": cite_id,
                    "type": "citation",
                    "group": "anchor",
                    "name": cite,
                    "comment_count": len(cids),
                })
            for cid in cids:
                edges.append({"source": cid, "target": cite_id, "type": "shared_citation", "weight": 1})

    stats = {
        "total_nodes": len(nodes),
        "comment_nodes": sum(1 for n in nodes if n["type"] == "comment"),
        "org_nodes": sum(1 for n in nodes if n["group"] == "orchestrator"),
        "citation_nodes": sum(1 for n in nodes if n["type"] == "citation"),
        "total_edges": len(edges),
    }
    return {"nodes": nodes, "edges": edges, "stats": stats}
