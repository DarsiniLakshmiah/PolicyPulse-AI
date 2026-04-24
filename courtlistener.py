"""
CourtListener integration — fetches real APA court cases relevant to a docket.
Requires a free token from courtlistener.com for live search.
Falls back to curated landmark APA cases when no token is set.

Get a free token: https://www.courtlistener.com/sign-in/ → Profile → API Token
Then add to .env:  COURTLISTENER_TOKEN=your-token-here
"""
import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

BASE  = "https://www.courtlistener.com/api/rest/v3"
TOKEN = os.getenv("COURTLISTENER_TOKEN", "").strip()

# Landmark APA cases always relevant to federal rulemaking challenges.
# Used as fallback when no CourtListener token is set.
LANDMARK_CASES = [
    {
        "case_name": "West Virginia v. EPA, 597 U.S. 697 (2022)",
        "court":     "Supreme Court of the United States",
        "date":      "2022-06-30",
        "snippet":   "Major questions doctrine: agencies must have clear congressional authorization "
                     "for rules of vast economic and political significance.",
        "url":       "https://www.courtlistener.com/opinion/8345822/west-virginia-v-environmental-protection-agency/",
    },
    {
        "case_name": "Motor Vehicle Mfrs. Ass'n v. State Farm, 463 U.S. 29 (1983)",
        "court":     "Supreme Court of the United States",
        "date":      "1983-06-24",
        "snippet":   "Foundational arbitrary-and-capricious standard: agency must examine relevant data "
                     "and articulate a satisfactory explanation for its action.",
        "url":       "https://www.courtlistener.com/opinion/111140/motor-vehicle-mfrs-assn-of-united-states-inc-v-state-farm-mut-automobile-ins-co/",
    },
    {
        "case_name": "Loper Bright Enterprises v. Raimondo, 603 U.S. ___ (2024)",
        "court":     "Supreme Court of the United States",
        "date":      "2024-06-28",
        "snippet":   "Overruled Chevron deference: courts must exercise independent judgment on "
                     "statutory interpretation rather than deferring to agency interpretations.",
        "url":       "https://www.courtlistener.com/opinion/10671965/loper-bright-enterprises-v-raimondo/",
    },
    {
        "case_name": "Ohio v. EPA, 603 U.S. ___ (2024)",
        "court":     "Supreme Court of the United States",
        "date":      "2024-06-27",
        "snippet":   "EPA rule stayed under APA arbitrary-and-capricious review; agency failed to "
                     "respond to significant comments raising serious objections.",
        "url":       "https://www.courtlistener.com/opinion/10671964/ohio-v-environmental-protection-agency/",
    },
    {
        "case_name": "FTC v. Actavis, Inc., 570 U.S. 136 (2013)",
        "court":     "Supreme Court of the United States",
        "date":      "2013-06-17",
        "snippet":   "Affirmed FTC's authority to challenge anti-competitive agreements; "
                     "relevant to scope of FTC Act Section 5 rulemaking authority.",
        "url":       "https://www.courtlistener.com/opinion/2104264/ftc-v-actavis-inc/",
    },
]


def _headers() -> dict:
    return {"Authorization": f"Token {TOKEN}"} if TOKEN else {}


def fetch_related_cases(query: str, max_results: int = 5) -> list:
    """
    Search CourtListener for opinions matching the query.
    Returns live results if COURTLISTENER_TOKEN is set,
    otherwise returns curated landmark cases.
    """
    if not TOKEN:
        return LANDMARK_CASES[:max_results]

    try:
        r = requests.get(
            f"{BASE}/search/",
            params={
                "q":              query,
                "type":           "o",
                "stat_Published": "on",
                "order_by":       "score desc",
                "format":         "json",
            },
            headers=_headers(),
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return LANDMARK_CASES[:max_results]

        cases = []
        for c in results[:max_results]:
            cases.append({
                "case_name": c.get("caseName", "Unknown"),
                "court":     c.get("court", ""),
                "date":      (c.get("dateFiled") or "")[:10],
                "snippet":   _clean(c.get("snippet", ""), 250),
                "url":       f"https://www.courtlistener.com{c.get('absolute_url', '')}",
            })
        return cases

    except Exception:
        return LANDMARK_CASES[:max_results]


def fetch_agency_cases(agency: str, keywords: str, max_results: int = 5) -> list:
    """
    Fetch cases relevant to a specific agency and rule topic.
    E.g. agency='FTC', keywords='Non-Compete Clause Ban'
    """
    query = f"{agency} {keywords} arbitrary capricious APA"
    return fetch_related_cases(query, max_results)


def _clean(text: str, max_len: int) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len] + ("…" if len(text) > max_len else "")
