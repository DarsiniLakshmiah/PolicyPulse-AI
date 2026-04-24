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

GEMINI_KEY    = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_KEY      = os.getenv("GROQ_API_KEY", "").strip()
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
GEMINI_MODEL  = "gemini-2.0-flash"
GROQ_MODEL    = "llama-3.3-70b-versatile"
CLAUDE_MODEL  = "claude-sonnet-4-6"


def active_llm() -> str:
    """Return which LLM is active."""
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
    Raises RuntimeError if no key is available.
    """
    if GEMINI_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel(
            GEMINI_MODEL,
            generation_config=genai.GenerationConfig(max_output_tokens=max_tokens, temperature=0.3),
        )
        resp = model.generate_content(prompt)
        return resp.text.strip()

    if GROQ_KEY:
        from groq import Groq
        client = Groq(api_key=GROQ_KEY)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    if ANTHROPIC_KEY:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    raise RuntimeError(
        "No LLM key found. Add GEMINI_API_KEY, GROQ_API_KEY, or ANTHROPIC_API_KEY to .env"
    )


def summarize_policy(title: str, text: str) -> str:
    """Plain-language summary of the proposed rule."""
    prompt = (
        f"Summarize this proposed federal rule in 2 sentences at a grade 8 reading level.\n"
        f"Rule: {title}\nText: {text[:500]}\n"
        f"Respond with just the summary, no preamble."
    )
    return call_llm(prompt, max_tokens=150)


def cluster_and_name(comment_texts: list, policy_title: str) -> list:
    """
    TF-IDF KMeans clustering, then LLM names each cluster.
    Returns list of dicts: {name, indices, count}.
    """
    if len(comment_texts) < 3:
        return [{"name": "General comments", "indices": list(range(len(comment_texts))), "count": len(comment_texts)}]

    n_clusters = min(4, len(comment_texts))
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    X = vectorizer.fit_transform(comment_texts)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    clusters: dict = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)

    named = []
    for label, indices in clusters.items():
        sample = " | ".join([comment_texts[i][:100] for i in indices[:3]])
        prompt = (
            f'These public comments on "{policy_title}" share a common theme.\n'
            f"Comments: {sample}\n"
            f"Give this cluster a name in 4-6 words. Respond with ONLY the name."
        )
        try:
            name = call_llm(prompt, max_tokens=20)
        except Exception:
            name = f"Theme {label + 1}"
        named.append({"name": name, "indices": indices, "count": len(indices)})

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
