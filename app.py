"""
PolicyPulse AI — Minimal Flask server. 3 routes only.
Run: python app.py
API at http://localhost:5000
"""
import os
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from ml_scorer import score_comment, detect_form_letter
from response_gen import summarize_policy, cluster_and_name, draft_response, generate_ogc_memo, active_llm
from courtlistener import fetch_agency_cases

load_dotenv()

app = Flask(__name__)
CORS(app)

REGULATIONS_API = "https://api.regulations.gov/v4"
# Use real key from .env if provided, otherwise free DEMO_KEY
API_KEY = os.getenv("REGULATIONS_GOV_API_KEY", "").strip() or "DEMO_KEY"

# In-memory session state (demo only — reset on server restart)
session_data = {
    "policy": None,
    "comments": [],
    "scored_comments": [],
    "clusters": [],
    "ogc_memo": "",
    "court_cases": [],
}

DEMO_COMMENTS = [
    {
        "id": "C-001",
        "name": "Patricia Webb, HR Director",
        "text": (
            "This rule would severely damage our ability to protect proprietary training investments. "
            "We spend $60,000 per technician. Without non-compete protections, competitors free-ride "
            "on our investments. We cite the Commission's own RFA analysis under 5 U.S.C. §604 which "
            "understates small business impact. We reserve all rights under the APA to challenge this rule."
        ),
    },
    {
        "id": "C-002",
        "name": "Carlos Reyes, Software Engineer",
        "text": (
            "I couldn't accept a $30,000 raise because of a non-compete. These clauses trap workers "
            "at below-market wages. The proposed rule correctly targets wage suppression. I support "
            "full implementation."
        ),
    },
    {
        "id": "C-003",
        "name": "United Workers Alliance",
        "text": "I support this rule. Workers deserve freedom to change jobs.",
    },
    {
        "id": "C-004",
        "name": "United Workers Alliance",
        "text": "I support this rule. Workers deserve freedom to change jobs.",
    },
    {
        "id": "C-005",
        "name": "Dr. Anika Sharma, Nurse Practitioner",
        "text": (
            "Healthcare non-competes created staffing deserts. I am subject to a 50-mile non-compete "
            "preventing practice in three HRSA-designated shortage areas in West Virginia. AMA's 2023 "
            "study documents 4,000 similar cases. This is a public health emergency requiring "
            "immediate regulatory action under CFR §910.3."
        ),
    },
    {
        "id": "C-006",
        "name": "Harrington & Cole LLP — Business Roundtable",
        "text": (
            "The Commission lacks authority under FTC Act Section 6(g). See West Virginia v. EPA, "
            "597 U.S. 697 (2022) — major questions doctrine applies. The economic analysis fails "
            "Executive Order 12866 requirements. We will seek judicial review under 5 U.S.C. §706 "
            "in the Fifth Circuit."
        ),
    },
    {
        "id": "C-007",
        "name": "James Thornton, IP Attorney",
        "text": (
            "The rule conflates non-competes with NDAs. Narrow §910.1 to exclude agreements "
            "protecting trade secrets under the Defend Trade Secrets Act, 18 U.S.C. §1836. "
            "A $100k salary threshold would protect workers without eliminating legitimate "
            "protections for senior executives."
        ),
    },
    {
        "id": "C-008",
        "name": "U.S. Chamber of Commerce",
        "text": (
            "The rule is arbitrary and capricious under 5 U.S.C. §706(2)(A). Motor Vehicle Mfrs. "
            "Ass'n v. State Farm, 463 U.S. 29 (1983). The Commission's own analysis acknowledges "
            "causal uncertainty yet proceeds to a blanket ban. The Chamber will seek injunctive "
            "relief in the Fifth Circuit if finalized."
        ),
    },
]

PRESETS = {
    "FTC-2023-0007": ("FTC Proposed Rule: Non-Compete Clause Ban", "FTC"),
    "EPA-HQ-OAR-2023-0072": ("EPA Vehicle Emissions Standards 2026", "EPA"),
    "CFPB-2023-0047": ("CFPB Medical Debt Credit Reporting Rule", "CFPB"),
}


@app.route("/api/import", methods=["POST"])
def import_rule():
    """Pull rule metadata from Regulations.gov (or preset), load comments."""
    body = request.get_json(force=True)
    docket_id = body.get("docketId", "FTC-2023-0007").strip()

    # Try live Regulations.gov API
    title, agency = PRESETS.get(docket_id, (docket_id, "Federal Agency"))
    try:
        r = requests.get(
            f"{REGULATIONS_API}/dockets/{docket_id}",
            params={"api_key": API_KEY},
            timeout=8,
        )
        r.raise_for_status()
        attrs = r.json().get("data", {}).get("attributes", {})
        title = attrs.get("title", title)
        agency = attrs.get("agencyId", agency)
    except Exception:
        pass  # use preset values

    # Step 1 — fetch comment IDs from listing (text not included in listing)
    comments = []
    try:
        r = requests.get(
            f"{REGULATIONS_API}/comments",
            params={
                "filter[docketId]": docket_id,
                "api_key": API_KEY,
                "page[size]": 10,
                "sort": "lastModifiedDate,documentId",
            },
            timeout=10,
        )
        r.raise_for_status()
        comment_ids = [item["id"] for item in r.json().get("data", [])]

        # Step 2 — fetch each comment individually to get full text
        for cid in comment_ids:
            try:
                rc = requests.get(
                    f"{REGULATIONS_API}/comments/{cid}",
                    params={"api_key": API_KEY},
                    timeout=8,
                )
                rc.raise_for_status()
                attrs = rc.json().get("data", {}).get("attributes", {})
                text = attrs.get("comment", "").strip()
                if not text or len(text) < 20:
                    continue
                first = attrs.get("firstName", "")
                last  = attrs.get("lastName", "")
                org   = attrs.get("organization", "")
                name  = f"{first} {last}".strip() or org or "Public Commenter"
                if org and org not in name:
                    name = f"{name}, {org}".strip(", ")
                comments.append({
                    "id":   cid,
                    "name": name,
                    "text": text,
                    "date": (attrs.get("postedDate") or "")[:10],
                    "duplicate_count": attrs.get("duplicateComments", 0),
                })
            except Exception:
                continue
    except Exception:
        pass

    # Always supplement with demo comments for a rich demo
    if len(comments) < 4:
        comments = DEMO_COMMENTS

    # Plain-language summary via Claude
    raw_text = f"Proposed rule: {title}. Agency: {agency}. Docket: {docket_id}."
    try:
        summary = summarize_policy(title, raw_text)
    except Exception:
        summary = f"This proposed rule from {agency} addresses federal regulatory requirements in docket {docket_id}."

    # Fetch related APA court cases from CourtListener
    court_cases = fetch_agency_cases(agency, title)

    session_data["policy"] = {
        "title": title,
        "agency": agency,
        "docket": docket_id,
        "summary": summary,
    }
    session_data["comments"] = comments
    session_data["scored_comments"] = []
    session_data["clusters"] = []
    session_data["ogc_memo"] = ""
    session_data["court_cases"] = court_cases

    return jsonify({
        "policy": session_data["policy"],
        "comment_count": len(comments),
        "court_cases": court_cases,
    })


@app.route("/api/process", methods=["POST"])
def process_comments():
    """
    Run full ML pipeline:
    1. ML significance scoring (trained classifier)
    2. Litigation risk scoring (rule-based)
    3. Form letter detection
    4. CFR section extraction
    5. TF-IDF KMeans clustering + Claude cluster naming
    6. Claude draft preamble responses per cluster
    7. OGC memo for high-risk comments
    """
    comments = session_data.get("comments", [])
    policy = session_data.get("policy", {})

    if not comments:
        return jsonify({"error": "No comments loaded. Call /api/import first."}), 400

    # Steps 1–4: ML scoring
    scored = [score_comment(c) for c in comments]
    scored = detect_form_letter(scored)

    # Step 5: Cluster significant, non-form-letter comments
    sig_comments = [
        c for c in scored
        if c["significant"] and not c["form_letter"]["is_form_letter"]
    ]
    texts = [c["text"] for c in sig_comments]

    try:
        clusters = cluster_and_name(texts, policy.get("title", ""))
    except Exception:
        clusters = [{
            "name": "Substantive Comments",
            "indices": list(range(len(texts))),
            "count": len(texts),
        }]

    # Step 6: Draft Claude response per cluster
    full_clusters = []
    for cl in clusters:
        samples = [texts[i] for i in cl["indices"][:3]]
        comment_ids = [sig_comments[i]["id"] for i in cl["indices"]]
        try:
            draft = draft_response(cl["name"], samples, policy.get("title", ""))
        except Exception:
            draft = (
                f"The agency has carefully considered the comments addressing "
                f"{cl['name']} and responds as follows in accordance with APA requirements..."
            )
        full_clusters.append({
            **cl,
            "comment_ids": comment_ids,
            "draft_response": draft,
        })

    # Step 7: OGC memo — enriched with real CourtListener cases
    high_risk = [c for c in scored if c["litigation_risk"] == "high"]
    court_cases = session_data.get("court_cases", [])
    try:
        ogc_memo = (
            generate_ogc_memo(high_risk, policy.get("title", ""), court_cases)
            if high_risk
            else "No high-risk comments identified in this docket."
        )
    except Exception:
        ogc_memo = "OGC review pending — LLM API unavailable."

    session_data["scored_comments"] = scored
    session_data["clusters"] = full_clusters
    session_data["ogc_memo"] = ogc_memo

    stats = {
        "total": len(scored),
        "significant": sum(1 for c in scored if c["significant"]),
        "form_letters": sum(1 for c in scored if c["form_letter"]["is_form_letter"]),
        "litigation_high": len(high_risk),
        "clusters": len(full_clusters),
    }

    return jsonify({
        "stats": stats,
        "comments": scored,
        "clusters": full_clusters,
        "ogc_memo": ogc_memo,
    })


@app.route("/api/upload", methods=["POST"])
def upload_comments():
    """
    Parse uploaded CSV or TXT file and load comments into session.

    CSV columns detected automatically (case-insensitive):
      text  : comment, text, body, comment_text
      name  : name, commenter, first_name + last_name, organization
      id    : id, document_id, comment_id, objectid
    TXT: split by blank line (paragraphs); fallback to one line each.
    """
    import csv
    import io

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    filename = f.filename or "upload"
    content = f.read().decode("utf-8-sig", errors="replace")  # strip BOM

    comments = []

    if filename.lower().endswith(".csv"):
        reader = csv.DictReader(io.StringIO(content))
        raw_headers = reader.fieldnames or []
        headers = [h.lower().strip() for h in raw_headers]
        header_map = {h.lower().strip(): h for h in raw_headers}

        def pick(candidates):
            for c in candidates:
                if c in headers:
                    return header_map[c]
            return None

        text_col  = pick(["comment", "text", "body", "comment_text", "comment text"])
        name_col  = pick(["name", "commenter", "commenter_name", "full_name", "author"])
        fname_col = pick(["first_name", "firstname", "first name"])
        lname_col = pick(["last_name", "lastname", "last name"])
        id_col    = pick(["id", "document_id", "comment_id", "objectid"])
        org_col   = pick(["organization", "org", "company"])

        for i, row in enumerate(reader):
            text = row.get(text_col, "").strip() if text_col else ""
            if not text or len(text) < 5:
                # fallback: first non-empty column value
                text = next((v.strip() for v in row.values() if v and len(v.strip()) > 5), "")
            if len(text) < 5:
                continue

            name = ""
            if name_col:
                name = row.get(name_col, "").strip()
            elif fname_col or lname_col:
                name = f"{row.get(fname_col or '', '').strip()} {row.get(lname_col or '', '').strip()}".strip()
            if org_col:
                org = row.get(org_col, "").strip()
                if org and org not in name:
                    name = f"{name}, {org}".strip(", ") if name else org
            name = name or "Public Commenter"

            comment_id = row.get(id_col, f"U-{i+1:03d}").strip() if id_col else f"U-{i+1:03d}"
            comments.append({"id": comment_id, "name": name, "text": text})

    else:  # TXT — split by paragraph, fallback to lines
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 10]
        if len(paragraphs) < 2:
            paragraphs = [ln.strip() for ln in content.splitlines() if len(ln.strip()) > 10]
        for i, para in enumerate(paragraphs):
            comments.append({"id": f"U-{i+1:03d}", "name": "Public Commenter", "text": para})

    if not comments:
        return jsonify({"error": "No valid comments found. CSV needs a 'comment' or 'text' column."}), 400

    if not session_data.get("policy"):
        session_data["policy"] = {
            "title": f"Uploaded: {filename}",
            "agency": "Uploaded Dataset",
            "docket": filename,
            "summary": f"{len(comments)} comments loaded from {filename}.",
        }

    session_data["comments"] = comments
    session_data["scored_comments"] = []
    session_data["clusters"] = []
    session_data["ogc_memo"] = ""

    return jsonify({
        "policy": session_data["policy"],
        "comment_count": len(comments),
        "court_cases": [],
    })


@app.route("/api/network", methods=["POST"])
def network():
    """Build comment coordination network graph from scored comments."""
    from ml_scorer import build_comment_network

    scored = session_data.get("scored_comments", [])
    if not scored:
        return jsonify({"error": "No scored comments. Run /api/process first."}), 400

    return jsonify(build_comment_network(scored))


@app.route("/api/apa-check", methods=["POST"])
def apa_check():
    """
    APA compliance check for a draft response.
    mode=rule  — instant regex check (default)
    mode=ai    — LLM self-evaluation merged over rule results
    """
    from apa_checker import check_apa, compute_score

    body = request.get_json(force=True)
    draft = body.get("draft", "")
    mode = body.get("mode", "rule")

    result = check_apa(draft)

    if mode == "ai" and draft and len(draft.strip()) >= 30:
        try:
            from response_gen import evaluate_apa_compliance
            ai_criteria = evaluate_apa_compliance(draft)
            ai_map = {c["id"]: c for c in ai_criteria}
            for criterion in result["criteria"]:
                ai = ai_map.get(criterion["id"])
                if ai:
                    criterion["status"] = ai.get("status", criterion["status"])
                    criterion["note"] = ai.get("note", "")
                    criterion["source"] = "ai"
            result["score"] = compute_score(result["criteria"])
            s = result["score"]
            result["level"] = "compliant" if s >= 80 else "partial" if s >= 50 else "deficient"
        except Exception as e:
            result["ai_error"] = str(e)

    return jsonify(result)


@app.route("/api/enrich", methods=["POST"])
def enrich_orgs():
    """
    Enrich high-significance commenter orgs via Crustdata.
    Only enriches orgs from comments with significance_score >= threshold (default 60).
    Results stored in session and returned for the frontend.
    """
    from crustdata import enrich_scored_comments, _token

    if not _token():
        return jsonify({"error": "CRUSTDATA_API_KEY not set in .env"}), 400

    scored = session_data.get("scored_comments", [])
    if not scored:
        return jsonify({"error": "Run /api/process first."}), 400

    body = request.get_json(force=True)
    threshold = int(body.get("threshold", 60))

    enriched = enrich_scored_comments(scored, sig_threshold=threshold)
    session_data["enriched_orgs"] = enriched

    return jsonify({
        "enriched_count": len(enriched),
        "orgs": enriched,
    })


@app.route("/api/health", methods=["GET"])
def health():
    """Health check — confirms ML model is loaded and shows active LLM."""
    model_exists = os.path.exists("model/significance_clf.pkl")
    llm = active_llm()
    return jsonify({
        "status": "ok",
        "model_loaded": model_exists,
        "llm": llm,
        "regulations_gov_key": "custom" if API_KEY != "DEMO_KEY" else "DEMO_KEY",
        "courtlistener": "authenticated" if os.getenv("COURTLISTENER_TOKEN", "").strip() else "public",
        "crustdata": "configured" if os.getenv("CRUSTDATA_API_KEY", "").strip() else "not configured",
        "message": "Run python train.py if model_loaded is false",
    })


if __name__ == "__main__":
    print("PolicyPulse AI — Flask backend")
    print(f"  LLM:            {active_llm()}")
    print(f"  Regulations.gov: {API_KEY}")
    print(f"  CourtListener:  {'authenticated' if os.getenv('COURTLISTENER_TOKEN','').strip() else 'public (no token)'}")
    print("  Health: http://localhost:5000/api/health")
    app.run(debug=True, port=5000)
