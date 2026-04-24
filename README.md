# PolicyPulse AI

> ML-powered federal comment analysis for regulatory attorneys and policy teams.

---

## The Problem

Every time a U.S. federal agency proposes a new rule, it must open a public comment period under the Administrative Procedure Act (APA). Major rulemakings routinely attract tens of thousands of comments — the FTC's non-compete ban drew over 26,000. Agencies are legally required to read and respond to every substantive comment in the final rule preamble. Miss one, and the rule is vulnerable to reversal in federal court.

The challenge is brutal in practice:

- **Volume**: A single rule can generate 10,000–100,000 comments in 60 days.
- **Signal-to-noise**: 95%+ of comments are short opinion statements ("I support this" / "Ban this now"). The 5% that cite statutes, challenge cost-benefit methodology, or threaten APA litigation are the ones that can overturn a rule — and they're buried in the pile.
- **Legal exposure**: Agencies that fail to respond to a substantive comment risk "arbitrary and capricious" challenges under 5 U.S.C. §706. Courts have vacated rules for exactly this reason.
- **Coordination blindness**: Law firms and industry groups orchestrate comment campaigns that look like grassroots opposition. Agencies have no systematic way to detect this.
- **Drafting burden**: Writing legally defensible preamble responses for 40+ comment themes, each requiring citation of statutory authority, takes months of attorney time.

The result: regulatory teams are overwhelmed, important comments slip through, and rules get challenged in court on procedural grounds that better tooling would have caught.

---

## The Solution

PolicyPulse AI is an end-to-end comment analysis platform that processes a federal docket in minutes and surfaces what matters.

**What it does:**

1. **Pulls live comments** from Regulations.gov by docket ID, or accepts CSV/TXT uploads of exported comment data.
2. **Scores every comment** with a trained ML classifier (TF-IDF + Logistic Regression) that distinguishes substantive legal arguments from opinion statements — trained on real federal comment data, not prompts.
3. **Detects litigation risk** using rule-based pattern matching on APA §706 language, statute citations, court references, and explicit legal threats.
4. **Detects form letter campaigns** by fingerprinting near-identical comment text, so agencies can weight coordinated campaigns appropriately.
5. **Maps comments to CFR sections** they reference, showing which regulatory provisions are most contested.
6. **Clusters themes** using TF-IDF KMeans and names each cluster via LLM.
7. **Drafts APA-compliant preamble responses** per cluster in Federal Register language, with a live compliance checker showing whether each draft satisfies the five APA §553 response criteria.
8. **Generates an OGC memo** (Office of General Counsel) summarizing litigation exposure, citing real case precedents pulled from CourtListener.
9. **Visualizes a coordination network** (D3.js force-directed graph) showing which law firms, advocacy orgs, and industry groups are connected through shared citations and form letter campaigns.
10. **Enriches org nodes** with live firmographic data (headcount, funding, HQ, job openings) via Crustdata — so agencies know whether a commenter is a 5-person advocacy shop or a Fortune 500 backed by $2B in VC.

### Data Sources

Every capability above is powered by real, publicly available federal data — not synthetic datasets or mock APIs.

| Data Source | What We Use It For | Access |
|---|---|---|
| **[Regulations.gov API v4](https://open.gsa.gov/api/regulationsgov/)** | Live comment fetch by docket ID; training corpus for the ML classifier | Free API key |
| **[Federal Register API](https://www.federalregister.gov/developers/documentation/api/v1)** | Final rule preamble text used to auto-label training data (comments cited in a preamble = significant) | Public, no key |
| **[CourtListener API](https://www.courtlistener.com/api/)** | Real APA precedent cases cited in the OGC memo (e.g. *Motor Vehicle Mfrs. v. State Farm*, *West Virginia v. EPA*) | Free with token |
| **[Crustdata /screener/company](https://crustdata.com)** | Firmographic enrichment for commenter orgs — headcount, funding, HQ, industry, job openings | Paid API |
| **Real federal dockets (training)** | Labeled comment pairs from `EPA-HQ-OAR-2021-0257`, `FTC-2023-0007`, `CFPB-2023-0047` used to train the significance classifier | Via Regulations.gov |

The ML model's ground truth is derived directly from agency behavior: if an agency cited a comment ID in its final rule preamble, that comment is labeled significant. No human labeling, no heuristics — the agency's own published response is the annotation.

---

## How It Differs from Existing Solutions

| | PolicyPulse AI | Quorum / FiscalNote | Docket Alarm | Manual Review |
|---|---|---|---|---|
| **ML significance scoring** | Trained on real Regulations.gov data | Keyword/topic tagging | Search & alerts only | Human judgment |
| **Litigation risk detection** | APA §706 pattern matching + case law | None | None | Attorney review |
| **Draft preamble responses** | LLM-generated, APA-checked | None | None | Attorney drafting |
| **Coordination network graph** | D3 force-directed, enriched org nodes | Stakeholder mapping | None | None |
| **Org firmographics** | Crustdata (headcount, funding, HQ) | Partial | None | None |
| **APA compliance checker** | Live, per-draft, rule-based + AI modes | None | None | None |
| **OGC memo generation** | Automated with real CourtListener citations | None | None | None |
| **Cost** | Open source | $$$$ enterprise contracts | $$ per search | Staff hours |

**The core architectural difference**: significance scoring is a trained ML classifier on real federal data — not a prompt, not keyword matching. The LLM only handles language generation tasks (naming clusters, drafting responses, writing memos). That separation is what makes the scoring reproducible, auditable, and fast.

Quorum and FiscalNote are excellent legislative tracking platforms built for lobbyists monitoring bill status, not agencies processing comment records. Docket Alarm is a legal research tool for finding documents, not analyzing their content. Neither produces draft regulatory responses or identifies litigation exposure.

---

## Target Audiences

**Primary**
- **Federal agency regulatory staff** — attorneys, policy analysts, and program officers who manage notice-and-comment rulemaking at agencies like EPA, FTC, CFPB, OSHA, FDA, and USDA.
- **Agency Offices of General Counsel** — who review litigation risk before a rule is finalized and need a prioritized view of which comments require a legal response.

**Secondary**
- **Regulatory law firms** — who represent clients submitting high-stakes comments and want to understand how their comment will be scored against the rest of the docket.
- **Policy research organizations** — think tanks and academic institutions studying public participation in rulemaking, comment effectiveness, and agency responsiveness.
- **Compliance teams at large enterprises** — who submit comments on rules affecting their industry and want to understand the significance and coverage of their submission.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Browser (HTML/JS)                         │
│  Tabs: Import · Comments · Responses · CFR Map · OGC · Network   │
│  D3.js network graph · Chart.js visualizations                   │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP (localhost:5000)
┌────────────────────────────▼─────────────────────────────────────┐
│                      Flask Backend (app.py)                      │
│                                                                  │
│  /api/import    — fetch docket from Regulations.gov              │
│  /api/upload    — parse uploaded CSV or TXT comment file         │
│  /api/process   — run full ML + LLM pipeline                     │
│  /api/enrich    — Crustdata org enrichment                       │
│  /api/network   — build coordination graph                       │
│  /api/apa-check — APA §553 compliance scoring                    │
│  /api/health    — status + active LLM                            │
└──────┬─────────────┬──────────────┬───────────────────────────────┘
       │             │              │
┌──────▼──────┐ ┌────▼──────┐ ┌────▼──────────────────────────────┐
│ ml_scorer   │ │ response_ │ │          External APIs             │
│ .py         │ │ gen.py    │ │                                    │
│             │ │           │ │  Regulations.gov — live comments   │
│ TF-IDF +    │ │ Gemini    │ │  CourtListener   — APA case law    │
│ Logistic    │ │ (primary) │ │  Crustdata       — org enrichment  │
│ Regression  │ │ Groq      │ │                                    │
│             │ │ (fallback)│ └────────────────────────────────────┘
│ Significance│ │ Claude    │
│ scoring     │ │ (fallback)│ ┌────────────────────────────────────┐
│             │ │           │ │         crustdata.py               │
│ Litigation  │ │ Cluster   │ │  POST /screener/company            │
│ risk        │ │ naming    │ │  Enriches high-sig org nodes only  │
│             │ │ Response  │ │  headcount · funding · HQ          │
│ Form letter │ │ drafting  │ │  job openings · industry           │
│ detection   │ │ OGC memo  │ └────────────────────────────────────┘
│             │ │ APA eval  │
│ Network     │ └───────────┘ ┌────────────────────────────────────┐
│ graph build │               │         apa_checker.py             │
└─────────────┘               │  Rule-based: regex patterns        │
                              │  AI mode: LLM self-evaluation      │
                              │  5 APA §553 criteria scored live   │
                              └────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                       train.py  (run once)                       │
│  Pulls comments from Regulations.gov training dockets            │
│  Auto-labels using final rule preamble text as ground truth      │
│  Trains TF-IDF + LogisticRegression pipeline                     │
│  Saves → model/significance_clf.pkl                              │
└──────────────────────────────────────────────────────────────────┘
```

### Request flow for a typical session

```
User enters docket ID
        ↓
/api/import  →  Regulations.gov API  (live comments + metadata)
        ↓
/api/process →  ml_scorer.score_comment()         [TF-IDF + LR]
             →  ml_scorer.detect_form_letter()    [text fingerprint]
             →  ml_scorer.extract_cfr_sections()  [regex]
             →  response_gen.cluster_and_name()   [KMeans + LLM]
             →  response_gen.draft_response()     [LLM per cluster]
             →  response_gen.generate_ogc_memo()  [LLM + CourtListener]
        ↓
/api/enrich  →  crustdata.enrich_scored_comments()
                (sig_score ≥ 60 only · org pattern detection)
        ↓
/api/network →  ml_scorer.build_comment_network()
                nodes: comments · orgs · citations · campaigns
                edges: authored_by · form_letter · shared_citation
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Vanilla HTML / CSS / JS | Single-file UI, zero build step |
| **Network graph** | D3.js v7 | Force-directed coordination graph |
| **Charts** | Chart.js v4 | Significance + CFR distribution |
| **Backend** | Python 3.11 + Flask | REST API server |
| **ML — Scoring** | scikit-learn (TF-IDF + LogisticRegression) | Comment significance classifier |
| **ML — Clustering** | scikit-learn (KMeans) | Theme grouping |
| **LLM — Primary** | Google Gemini 2.0 Flash | Cluster naming · response drafting · OGC memo |
| **LLM — Fallback** | Groq (Llama 3.3 70B) | Same tasks, fast inference |
| **LLM — Fallback** | Anthropic Claude Sonnet | Same tasks |
| **Org enrichment** | Crustdata `/screener/company` | Firmographics for commenter orgs |
| **Case law** | CourtListener API | Real APA precedents for OGC memo |
| **Comment data** | Regulations.gov API v4 | Live federal comment fetch |
| **Model persistence** | joblib | Trained classifier save/load |
| **Env management** | python-dotenv | API key loading |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/policypulse-ai.git
cd policypulse-ai
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # Mac / Linux
```

Edit `.env`:

```env
GEMINI_API_KEY=your_key            # required  — aistudio.google.com
REGULATIONS_GOV_API_KEY=your_key   # optional  — improves live comment fetch
COURTLISTENER_TOKEN=your_token     # optional  — authenticated case search
CRUSTDATA_API_KEY=your_key         # optional  — org enrichment
```

### 3. Train the ML model (one-time)

```bash
python train.py
```

Pulls real comments from Regulations.gov training dockets, auto-labels them using preamble text as ground truth, trains the classifier, and saves it to `model/significance_clf.pkl`.

### 4. Start the backend

```bash
python app.py
```

Verify at `http://localhost:5000/api/health` — should show `"model_loaded": true` and your active LLM.

### 5. Open the frontend

```bash
python -m http.server 8080
```

Open `http://localhost:8080/frontend/index.html`

---

## Usage

| Step | Where | What to do |
|---|---|---|
| 1 | Import tab | Enter a docket ID (`FTC-2023-0007`) or upload a CSV/TXT file |
| 2 | Import tab | Click **Run ML Analysis** |
| 3 | Comments tab | Filter by significance / litigation risk / form letters · export CSV |
| 4 | Responses tab | Review and edit AI-drafted preamble responses · check APA compliance |
| 5 | CFR Mapping tab | See which regulatory sections are most contested |
| 6 | OGC Risk tab | Review litigation risk scores · read generated OGC memo |
| 7 | Network tab | Explore coordination graph · click **Enrich Orgs** for firmographics |

---

## CSV Upload Format

```csv
id,name,organization,comment
C-001,Jane Smith,Harrington & Cole LLP,"This rule violates 5 U.S.C. §706..."
C-002,John Doe,,"I support this rule."
```

Recognized column names (case-insensitive): `comment` / `text` / `body` for the text; `name` / `commenter` / `first_name` + `last_name` for the author; `organization` / `org`; `id` / `document_id`.

---

## Project Structure

```
policypulse-ai/
├── app.py              # Flask server — all API routes
├── ml_scorer.py        # Classifier, litigation risk, network builder
├── response_gen.py     # LLM calls — Gemini / Groq / Claude
├── train.py            # Fetch real data + train + save model
├── crustdata.py        # Crustdata org enrichment client
├── apa_checker.py      # APA §553 compliance checker
├── courtlistener.py    # CourtListener API client
├── requirements.txt
├── .env                # API keys — gitignored
├── .gitignore
├── model/
│   └── significance_clf.pkl    # Trained classifier (from train.py)
├── data/
│   └── training_data.json      # Labeled training examples
└── frontend/
    └── index.html      # Single-file UI
```

---

## License

MIT
