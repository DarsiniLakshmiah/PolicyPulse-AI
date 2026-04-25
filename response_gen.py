"""
Language generation — Groq (Llama 3.1 70B) preferred, Claude fallback.
Reads keys from .env automatically.

Functions:
  summarize_policy     — plain-language rule summary
  cluster_and_name     — TF-IDF KMeans + LLM cluster naming
  draft_response       — Federal Register preamble response per cluster
  generate_ogc_memo    — OGC litigation risk memo with real case context
"""
import os
import re
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

load_dotenv()

OPENAI_KEY    = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_KEY    = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_KEY      = os.getenv("GROQ_API_KEY", "").strip()
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
OPENAI_MODEL  = "gpt-4o-mini"
GEMINI_MODEL  = "gemini-2.0-flash"
GROQ_MODEL    = "llama-3.3-70b-versatile"
CLAUDE_MODEL  = "claude-sonnet-4-6"


def active_llm() -> str:
    """Return which LLM is active."""
    if OPENAI_KEY:
        return f"openai/{OPENAI_MODEL}"
    if GEMINI_KEY:
        return f"gemini/{GEMINI_MODEL}"
    if GROQ_KEY:
        return f"groq/{GROQ_MODEL}"
    if ANTHROPIC_KEY:
        return f"anthropic/{CLAUDE_MODEL}"
    return "none"


def call_llm(prompt: str, max_tokens: int = 1000) -> str:
    """
    Priority: Gemini → Groq → Claude.
    Retries once on 429 rate-limit errors with the suggested backoff.
    Raises RuntimeError if no key is available.
    """
    import time

    def _openai():
        from openai import OpenAI
        resp = OpenAI(api_key=OPENAI_KEY).chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    def _gemini():
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel(
            GEMINI_MODEL,
            generation_config=genai.GenerationConfig(max_output_tokens=max_tokens, temperature=0.3),
        )
        return model.generate_content(prompt).text.strip()

    def _groq():
        from groq import Groq
        resp = Groq(api_key=GROQ_KEY).chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    def _claude():
        import anthropic
        resp = anthropic.Anthropic(api_key=ANTHROPIC_KEY).messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    providers = []
    if OPENAI_KEY:
        providers.append(("openai", _openai))
    if GEMINI_KEY:
        providers.append(("gemini", _gemini))
    if GROQ_KEY:
        providers.append(("groq", _groq))
    if ANTHROPIC_KEY:
        providers.append(("claude", _claude))

    if not providers:
        raise RuntimeError(
            "No LLM key found. Add GEMINI_API_KEY, GROQ_API_KEY, or ANTHROPIC_API_KEY to .env"
        )

    last_err = None
    for name, fn in providers:
        for attempt in range(2):
            try:
                return fn()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    # Parse suggested retry delay from error message, default 30s
                    import re as _re
                    m = _re.search(r"retry in ([\d.]+)s", err_str)
                    wait = float(m.group(1)) + 1 if m else 30
                    if attempt == 0:
                        time.sleep(min(wait, 60))
                        continue  # retry same provider
                last_err = e
                break  # move to next provider

    raise RuntimeError(f"All LLM providers failed. Last error: {last_err}")


def summarize_policy(title: str, text: str) -> str:
    """Plain-language summary of the proposed rule."""
    prompt = (
        f"Summarize this proposed federal rule in 2 sentences at a grade 8 reading level.\n"
        f"Rule: {title}\nText: {text[:500]}\n"
        f"Respond with just the summary, no preamble."
    )
    return call_llm(prompt, max_tokens=150)


def _keyword_cluster_name(texts: list) -> str:
    """Extract top keywords from a cluster as a readable fallback name."""
    from sklearn.feature_extraction.text import TfidfVectorizer as _TV
    try:
        v = _TV(max_features=200, stop_words="english", ngram_range=(1, 2))
        X = v.fit_transform(texts)
        scores = X.sum(axis=0).A1
        top = sorted(zip(v.get_feature_names_out(), scores), key=lambda x: -x[1])
        keywords = [w for w, _ in top[:3] if len(w) > 3]
        return " · ".join(keywords).title() if keywords else "Substantive Comments"
    except Exception:
        return "Substantive Comments"


def cluster_and_name(comment_texts: list, policy_title: str) -> list:
    """
    TF-IDF KMeans clustering, then one batched LLM call names all clusters.
    Falls back to keyword extraction if LLM is unavailable.
    Returns list of dicts: {name, indices, count}.
    """
    if len(comment_texts) < 3:
        name = _keyword_cluster_name(comment_texts)
        return [{"name": name, "indices": list(range(len(comment_texts))), "count": len(comment_texts)}]

    n_clusters = min(4, len(comment_texts))
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    X = vectorizer.fit_transform(comment_texts)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    clusters: dict = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)

    # Build keyword fallback names first (always available)
    keyword_names = {
        label: _keyword_cluster_name([comment_texts[i] for i in indices])
        for label, indices in clusters.items()
    }

    # One batched LLM call for all cluster names
    cluster_names = dict(keyword_names)  # start with keyword fallbacks
    try:
        sections = []
        for label, indices in clusters.items():
            sample = " | ".join([comment_texts[i][:120] for i in indices[:3]])
            sections.append(f"Cluster {label}: {sample}")

        prompt = (
            f'You are naming comment clusters from a federal rulemaking on "{policy_title}".\n'
            f"For each cluster below, give a precise 4-6 word name describing the legal or policy theme.\n"
            f"Reply with ONLY a JSON object mapping cluster number to name. Example: {{\"0\": \"APA Procedural Challenges\", \"1\": \"Worker Wage Suppression\"}}\n\n"
            + "\n\n".join(sections)
        )
        raw = call_llm(prompt, max_tokens=120)
        # Parse JSON from response
        import json as _json
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            parsed = _json.loads(m.group())
            for label in clusters:
                key = str(label)
                if key in parsed and parsed[key].strip():
                    cluster_names[label] = parsed[key].strip()
    except Exception as e:
        import logging
        logging.warning("Cluster naming LLM call failed, using keywords: %s", e)

    named = []
    for label, indices in clusters.items():
        named.append({
            "name": cluster_names[label],
            "indices": indices,
            "count": len(indices),
        })
    return named


def draft_response(cluster_name: str, comment_samples: list, policy_title: str) -> str:
    """Draft an APA preamble response for a theme cluster."""
    samples = "\n".join([f"- {c[:200]}" for c in comment_samples[:3]])
    prompt = (
        f'You are a federal agency attorney drafting the official Response to Comments\n'
        f'for the final rule preamble of "{policy_title}".\n\n'
        f'Theme: "{cluster_name}"\n'
        f"Sample comments:\n{samples}\n\n"
        f"Write a 2-paragraph APA-compliant response in Federal Register language.\n"
        f"Paragraph 1: Acknowledge the comments and summarize the concern.\n"
        f"Paragraph 2: Agency position with legal and technical rationale.\n"
        f"Be formal, specific, and cite statutory authority where relevant."
    )
    return call_llm(prompt, max_tokens=500)


def evaluate_apa_compliance(draft: str) -> list:
    """
    LLM self-evaluation of APA compliance.
    Returns list of criteria dicts with status ('pass'|'warn'|'fail') and a note.
    """
    prompt = (
        "You are an APA §553 compliance reviewer for federal agency rulemaking.\n"
        "Evaluate this draft Response to Comments against the 5 criteria below.\n"
        "Reply ONLY with a valid JSON array — no prose, no markdown fences.\n\n"
        f"Draft:\n{draft[:2000]}\n\n"
        "Criteria (status must be exactly 'pass', 'warn', or 'fail'):\n"
        "1. acknowledged   — Did it explicitly acknowledge the commenters' concern?\n"
        "2. reasoned_basis — Did it provide substantive reasoning (not just a conclusion)?\n"
        "3. cited_authority — Did it cite a statute, regulation, case, or executive order?\n"
        "4. addressed_substance — Did it engage with the merits of the concern?\n"
        "5. formal_register — Is it in formal Federal Register preamble style?\n\n"
        'Return exactly this shape (one sentence for note):\n'
        '[{"id":"acknowledged","status":"pass","note":"..."},'
        '{"id":"reasoned_basis","status":"pass","note":"..."},'
        '{"id":"cited_authority","status":"fail","note":"..."},'
        '{"id":"addressed_substance","status":"warn","note":"..."},'
        '{"id":"formal_register","status":"pass","note":"..."}]'
    )
    import json
    raw = call_llm(prompt, max_tokens=400)
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def generate_ogc_memo(
    high_risk_comments: list,
    policy_title: str,
    court_cases: list = None,
) -> str:
    """
    Generate an OGC litigation risk memo for General Counsel.
    Incorporates real CourtListener cases when available.
    """
    comment_list = "\n".join([
        f"- {c.get('name', 'Commenter')}: {', '.join(c.get('litigation_flags', []))} "
        f"(litigation score: {c.get('litigation_score', 0)})"
        for c in high_risk_comments[:5]
    ])

    cases_section = ""
    if court_cases:
        case_lines = "\n".join([
            f"- {c['case_name']} ({c['court']}, {c['date'][:4] if c.get('date') else 'n.d.'})"
            for c in court_cases[:4]
        ])
        cases_section = f"\nRelated precedents from CourtListener:\n{case_lines}\n"

    prompt = (
        f'You are the Office of General Counsel reviewing litigation risk for "{policy_title}".\n\n'
        f"High-risk comments flagged by automated screening:\n{comment_list}\n"
        f"{cases_section}\n"
        f"Write a formal internal OGC memorandum with TO/FROM/DATE/RE headers, then numbered sections:\n"
        f"1. Overall litigation risk assessment (Low / Medium / High / Critical)\n"
        f"2. Primary legal vulnerabilities — reference specific patterns and any relevant precedents\n"
        f"3. Recommended actions for the response team before finalization\n"
        f"Use formal legal language. 3-4 paragraphs."
    )
    return call_llm(prompt, max_tokens=700)
