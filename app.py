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
    # ── Law firm comments (shared citations create citation nodes) ──
    {
        "id": "C-001",
        "name": "Harrington & Cole LLP — Business Roundtable",
        "text": (
            "The Commission lacks statutory authority under FTC Act Section 6(g) to issue this rule. "
            "Under the major questions doctrine articulated in West Virginia v. EPA, 597 U.S. 697 (2022), "
            "Congress must speak clearly before an agency can regulate a matter of vast economic significance. "
            "The economic analysis also fails Executive Order 12866 requirements. "
            "We will seek judicial review under 5 U.S.C. §706 in the Fifth Circuit if finalized."
        ),
    },
    {
        "id": "C-002",
        "name": "Gibson, Dunn & Crutcher LLP — U.S. Chamber of Commerce",
        "text": (
            "The rule is arbitrary and capricious under 5 U.S.C. §706(2)(A). Motor Vehicle Mfrs. "
            "Ass'n v. State Farm, 463 U.S. 29 (1983). The Commission's own analysis acknowledges "
            "causal uncertainty yet proceeds to a categorical ban without adequate justification. "
            "West Virginia v. EPA, 597 U.S. 697 (2022) controls. The Chamber will seek injunctive "
            "relief in the D.C. Circuit if finalized. We reserve all rights under the APA."
        ),
    },
    {
        "id": "C-003",
        "name": "Crowell & Moring LLP — National Federation of Independent Business",
        "text": (
            "The Commission's Regulatory Flexibility Act analysis under 5 U.S.C. §604 is fatally "
            "deficient. The RFA requires a genuine analysis of small business alternatives — not the "
            "cursory treatment provided here. Our survey of 1,847 NFIB members documents compliance "
            "costs 340% higher than the Commission's estimate. We will seek judicial review under "
            "5 U.S.C. §706 if the agency fails to correct this analysis before finalization."
        ),
    },
    # ── Industry groups ──
    {
        "id": "C-004",
        "name": "U.S. Chamber of Commerce",
        "text": (
            "The rule's definition in §910.1 is overbroad and captures agreements that protect "
            "legitimate trade secrets under the Defend Trade Secrets Act, 18 U.S.C. §1836. "
            "The Commission's own data shows non-competes are concentrated in high-wage, high-skill "
            "roles where employer investment is substantial. Executive Order 12866 requires the agency "
            "to demonstrate that benefits outweigh costs — a showing the NPRM fails to make."
        ),
    },
    {
        "id": "C-005",
        "name": "Business Roundtable",
        "text": (
            "Member companies have structured long-term R&D investment strategies around the "
            "enforceability of existing agreements. The Commission's failure to consider these reliance "
            "interests renders the rule arbitrary and capricious under Motor Vehicle Mfrs. Ass'n v. "
            "State Farm, 463 U.S. 29 (1983). We urge the agency to adopt a salary threshold of "
            "$150,000 as a less restrictive alternative consistent with Executive Order 12866."
        ),
    },
    {
        "id": "C-006",
        "name": "National Federation of Independent Business",
        "text": (
            "Small businesses with fewer than 50 employees represent 89% of firms using non-competes "
            "for workers earning under $50,000 per year per our survey of 1,200 members. The Commission's "
            "Regulatory Flexibility Act analysis under 5 U.S.C. §604 does not adequately address this. "
            "We request a 60-day extension of the comment period and a small business review panel."
        ),
    },
    # ── Labor organizations ──
    {
        "id": "C-007",
        "name": "AFL-CIO",
        "text": (
            "The AFL-CIO and its 12.5 million members strongly support the proposed rule. Non-compete "
            "agreements suppress wages, trap workers in unsafe conditions, and undermine collective "
            "bargaining. Our economic analysis shows workers subject to non-competes earn 4.3% less "
            "than comparable workers without them. The Commission has clear authority under FTC Act "
            "Section 5 to prohibit unfair methods of competition. We urge immediate finalization."
        ),
    },
    {
        "id": "C-008",
        "name": "Service Employees International Union — SEIU",
        "text": (
            "SEIU represents 2 million workers, many of whom are trapped by non-compete clauses in "
            "low-wage service jobs. Healthcare workers in particular face 50-mile non-competes that "
            "prevent them from leaving abusive employers. The Commission's proposed rule correctly "
            "identifies non-competes as an unfair method of competition. We support full implementation "
            "without a salary threshold carve-out."
        ),
    },
    # ── Advocacy organizations ──
    {
        "id": "C-009",
        "name": "Public Citizen",
        "text": (
            "Public Citizen submits these comments in strong support of the proposed rule. Non-compete "
            "clauses function as a tax on worker mobility that accrues entirely to employers. The "
            "Commission's economic analysis, drawing on Starr, Prescott & Bishara (2021), correctly "
            "identifies wage suppression as the primary harm. We urge the Commission to reject any "
            "salary threshold carve-out, which would undermine the rule's core protective purpose."
        ),
    },
    # ── Individual workers (personal stories, low litigation score) ──
    {
        "id": "C-010",
        "name": "Dr. Anika Sharma, Nurse Practitioner",
        "text": (
            "I am subject to a 50-mile non-compete preventing practice in three HRSA-designated "
            "Health Professional Shortage Areas in rural West Virginia. AMA's 2023 study documents "
            "4,000 similar cases nationwide. I have been unable to accept two positions at federally "
            "qualified health centers because of this clause. This is a patient safety emergency."
        ),
    },
    {
        "id": "C-011",
        "name": "Carlos Reyes, Software Engineer",
        "text": (
            "I turned down a $30,000 raise at a competing firm because of a non-compete I signed "
            "on my first day without reading carefully. These clauses trap workers at below-market "
            "wages and the company knows it. I fully support this rule and urge the Commission "
            "to finalize it without delay."
        ),
    },
    {
        "id": "C-012",
        "name": "Patricia Webb, Registered Nurse",
        "text": (
            "After 12 years at a hospital system, I was presented with a non-compete covering a "
            "30-mile radius for two years. When I left for a better job, I had to drive 45 minutes "
            "each way just to stay in my profession. My patients lost continuity of care. "
            "Please finalize this rule."
        ),
    },
    {
        "id": "C-013",
        "name": "Marcus Johnson, Restaurant Chef",
        "text": (
            "I signed a non-compete as a line cook making $14 an hour. I had no idea it would "
            "prevent me from working at any restaurant within 10 miles for a year. I had to take "
            "a job in retail to survive. This rule would protect workers like me who have no "
            "bargaining power when we sign these agreements."
        ),
    },
    # ── Form letter campaign A — worker support (4 identical) ──
    {
        "id": "C-014",
        "name": "Amanda Torres",
        "text": "I am writing in strong support of the FTC's proposed rule banning non-compete clauses. Non-competes hurt workers and limit economic freedom. Please finalize this rule without delay.",
    },
    {
        "id": "C-015",
        "name": "David Kim",
        "text": "I am writing in strong support of the FTC's proposed rule banning non-compete clauses. Non-competes hurt workers and limit economic freedom. Please finalize this rule without delay.",
    },
    {
        "id": "C-016",
        "name": "Sandra Okonkwo",
        "text": "I am writing in strong support of the FTC's proposed rule banning non-compete clauses. Non-competes hurt workers and limit economic freedom. Please finalize this rule without delay.",
    },
    {
        "id": "C-017",
        "name": "James Whitfield",
        "text": "I am writing in strong support of the FTC's proposed rule banning non-compete clauses. Non-competes hurt workers and limit economic freedom. Please finalize this rule without delay.",
    },
    # ── Form letter campaign B — employer opposition (3 identical) ──
    {
        "id": "C-018",
        "name": "Regional Employers Coalition",
        "text": "We oppose the FTC's proposed non-compete rule. This regulation will destroy employer investment in workforce training and harm American competitiveness. Please withdraw this misguided proposal.",
    },
    {
        "id": "C-019",
        "name": "Midwest Business Alliance",
        "text": "We oppose the FTC's proposed non-compete rule. This regulation will destroy employer investment in workforce training and harm American competitiveness. Please withdraw this misguided proposal.",
    },
    {
        "id": "C-020",
        "name": "Southeast Chamber Network",
        "text": "We oppose the FTC's proposed non-compete rule. This regulation will destroy employer investment in workforce training and harm American competitiveness. Please withdraw this misguided proposal.",
    },
    # ── Academic / expert comments ──
    {
        "id": "C-021",
        "name": "Prof. Elena Vasquez, Stanford Law School",
        "text": (
            "The Commission's reliance on Starr, Prescott & Bishara (2021) is methodologically sound "
            "but the wage effect estimate should be updated using Johnson & Lipsitz (2022), which "
            "finds stronger effects for workers below $60,000. The major questions doctrine under "
            "West Virginia v. EPA, 597 U.S. 697 (2022) does not apply here — non-compete policy "
            "is squarely within the FTC's traditional Section 5 authority."
        ),
    },
    {
        "id": "C-022",
        "name": "State of California — Office of the Attorney General",
        "text": (
            "The State of California submits these comments to note that California Business & "
            "Professions Code §16600 has prohibited non-competes since 1872 with no adverse effect "
            "on employer investment or innovation. California's tech sector demonstrates that "
            "strong worker mobility protections and business dynamism are complementary. "
            "The Commission should reject any state preemption carve-out in the final rule."
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
                first = attrs.get("firstName") or ""
                last  = attrs.get("lastName") or ""
                org   = attrs.get("organization") or ""
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
    # Fall back to ALL scored comments if ML marks nothing significant
    # (happens when live API returns only opinion-statement comments)
    if not sig_comments:
        sig_comments = [c for c in scored if not c["form_letter"]["is_form_letter"]]
    texts = [c["text"] for c in sig_comments]

    try:
        clusters = cluster_and_name(texts, policy.get("title", ""))
    except Exception as e:
        import logging
        logging.warning("cluster_and_name failed: %s", e)
        clusters = [{
            "name": "Substantive Comments",
            "indices": list(range(len(texts))),
            "count": len(texts),
        }]

    # Step 6: Draft response per cluster
    full_clusters = []
    for cl in clusters:
        samples = [texts[i] for i in cl["indices"][:3]]
        comment_ids = [sig_comments[i]["id"] for i in cl["indices"]]
        try:
            draft = draft_response(cl["name"], samples, policy.get("title", ""))
        except Exception as e:
            import logging
            logging.warning("draft_response failed for cluster '%s': %s", cl["name"], e)
            draft = f"[Draft generation failed: {e}. Check your GEMINI_API_KEY and retry.]"
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
