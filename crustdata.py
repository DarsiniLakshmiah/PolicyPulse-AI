"""
Crustdata org enrichment client.
POST /screener/company — returns firmographic data by company name.
Only called for high-significance commenter orgs to conserve credits.
"""
import os
import re
import time
import logging
import requests

logger = logging.getLogger(__name__)
CRUSTDATA_BASE = "https://api.crustdata.com"

_ORG_PATTERN = re.compile(
    r"(LLC|LLP|Inc\.?|Corp\.?|P\.C\.|Foundation|Alliance|Association|Society|"
    r"Chamber|Union|Institute|Council|Bureau|Coalition|Group|Center|Network|Fund|"
    r"University|College|Hospital|Health|Medical|Law|Capital|Partners|Advisory|"
    r"Roundtable|Federation|League|Conference|Committee|Board|Authority|"
    r"& Co|and Company|\bAMA\b|\bABA\b|\bNFIB\b|\bNAACP\b)",
    re.IGNORECASE,
)


def _token() -> str:
    return os.getenv("CRUSTDATA_API_KEY", "").strip()


def enrich_company(name: str) -> dict | None:
    """
    Enrich a single company by name via Crustdata.
    Returns normalised firmographic dict or None if not found / no key.
    """
    token = _token()
    if not token or not name:
        return None
    try:
        r = requests.post(
            f"{CRUSTDATA_BASE}/screener/company",
            headers={
                "Authorization": f"Token {token}",
                "Content-Type": "application/json",
            },
            json={
                "filters": [
                    {"filter_type": "company_name", "value": name, "operator": "equals"}
                ],
                "page": 1,
                "limit": 1,
            },
            timeout=12,
        )
        r.raise_for_status()
        rows = r.json().get("data", [])
        return _normalise(rows[0]) if rows else None
    except Exception as e:
        logger.warning("Crustdata enrichment failed for %r: %s", name, e)
        return None


def _normalise(raw: dict) -> dict:
    """Map Crustdata field names to our internal schema."""
    return {
        "headcount":         raw.get("headcount"),
        "headcount_growth":  raw.get("headcount_qoq_pct"),
        "total_funding_usd": raw.get("total_funding_raised_usd"),
        "last_round":        raw.get("last_funding_round_type"),
        "website":           raw.get("company_website_domain"),
        "linkedin":          raw.get("linkedin_profile_url"),
        "founded":           raw.get("founded_year"),
        "country":           raw.get("hq_country"),
        "city":              raw.get("hq_city"),
        "industry":          raw.get("industry"),
        "job_openings":      raw.get("job_openings_count"),
        "web_traffic":       raw.get("web_traffic_monthly"),
    }


def _extract_org(commenter_name: str) -> str | None:
    """
    Extract the org portion from a commenter name string.
    Handles patterns like:
      "Jane Smith, Harrington & Cole LLP"   → "Harrington & Cole LLP"
      "U.S. Chamber of Commerce"            → "U.S. Chamber of Commerce"
      "Jane Smith, Senior Attorney"         → None  (title, not org)
    """
    name = commenter_name.strip()
    if not name:
        return None

    # Split on " — " (e.g. "Harrington & Cole LLP — Business Roundtable")
    # and try each part
    parts = [p.strip() for p in re.split(r"\s+[—–-]{1,2}\s+", name)]

    candidates = []
    for part in parts:
        # If there's a comma, take either side that looks like an org
        subparts = [s.strip() for s in part.split(",", 1)]
        for sp in subparts:
            if sp and _ORG_PATTERN.search(sp):
                candidates.append(sp)
        # Also keep the whole part if it looks like an org
        if _ORG_PATTERN.search(part) and part not in candidates:
            candidates.append(part)

    return candidates[0] if candidates else None


def enrich_scored_comments(scored: list, sig_threshold: int = 60) -> dict:
    """
    Enrich unique orgs from comments with significance_score >= sig_threshold.
    Returns {org_name: enriched_dict}.  Skips individual names that don't
    match known org patterns to avoid wasting credits.
    """
    seen: set[str] = set()
    queue: list[str] = []

    for c in scored:
        if (c.get("significance_score") or 0) < sig_threshold:
            continue
        org = _extract_org(c.get("name") or "")
        if org and org not in seen:
            seen.add(org)
            queue.append(org)

    logger.info("Crustdata: enriching %d unique orgs (threshold=%d)", len(queue), sig_threshold)

    results: dict = {}
    for org in queue:
        data = enrich_company(org)
        if data:
            results[org] = data
        time.sleep(0.2)  # gentle rate limiting

    logger.info("Crustdata: %d/%d orgs enriched successfully", len(results), len(queue))
    return results
