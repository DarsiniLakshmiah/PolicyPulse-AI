"""
Rule-based APA §553 compliance checker for draft Response-to-Comments text.
Five criteria derived from Motor Vehicle Mfrs. v. State Farm and DC Circuit doctrine.
"""
import re

APA_CRITERIA = [
    {
        "id": "acknowledged",
        "label": "Acknowledged comment",
        "desc": "Explicitly acknowledges the commenter's concern",
        "patterns": [
            r"\b(comment(?:er)?s?)\b.{0,80}\b(raised|noted|expressed|argued|stated|submitted|contends?|asserts?)\b",
            r"\b(the agency|the commission|the bureau|we)\b.{0,50}\b(received|reviewed|considered|notes?|acknowledges?)\b",
            r"\b(this comment|these comments|the comments?\s+(?:addressed|regarding|concerning|about))\b",
            r"\b(several|many|some|a number of)\b.{0,40}\b(comments?|commenters?)\b",
        ],
    },
    {
        "id": "reasoned_basis",
        "label": "Reasoned explanation",
        "desc": "Provides substantive reasoning, not just a bare conclusion",
        "patterns": [
            r"\b(because|therefore|accordingly|thus|as a result|given that|in light of|for (?:these|this|the following) reasons?)\b",
            r"\b(evidence|data|studies?|analysis|research|record|demonstrates?|shows?|indicates?|establishes?)\b",
            r"\b(?:the agency|the commission|the bureau|we)\b.{0,40}\b(finds?|concludes?|determines?|believes?|considers?)\b",
            r"\b(upon (?:careful |further )?consideration|after (?:careful |thorough )?review)\b",
        ],
    },
    {
        "id": "cited_authority",
        "label": "Cited legal authority",
        "desc": "Cites a statute, regulation, case, or executive order",
        "patterns": [
            r"\d+\s+U\.S\.C\.?\s*§?\s*\d+",
            r"§\s*\d+\.\d+",
            r"\d+\s+C\.F\.R\.?\s*(?:[Pp]art\s*)?\d+",
            r"\b(?:pursuant to|under|authorized by|consistent with)\b.{0,80}\b(?:Act|statute|regulation|Section|Rule|Order)\b",
            r"\bExecutive Order\s+\d+",
            r"\d+\s+U\.S\.\s+\d+",
        ],
    },
    {
        "id": "addressed_substance",
        "label": "Addressed substance",
        "desc": "Engages with the merits, not just procedural points",
        "patterns": [
            r"\b(?:however|nevertheless|nonetheless|while|although|notwithstanding)\b",
            r"\b(?:the agency|the commission|the bureau|we)\b.{0,40}\b(?:disagrees?|agrees?|concurs?|declines?|adopts?|rejects?|accepts?|retains?|modifies?)\b",
            r"\b(?:this (?:concern|argument|assertion|claim|position|suggestion))\b",
            r"\b(?:agree|disagree|concur|reject|adopt|retain|decline)\b.{0,40}\b(?:the|this|these)\b",
        ],
    },
    {
        "id": "formal_register",
        "label": "Federal Register style",
        "desc": "Uses formal Federal Register preamble language",
        "patterns": [
            r"\b(?:the (?:final|proposed) rule)\b",
            r"\b(?:the agency|the commission|the bureau|the department)\b",
            r"\b(?:preamble|rulemaking|NPRM|notice of proposed rulemaking)\b",
            r"\bwe (?:are|have) (?:finaliz|adopt|revis|amend|retain|modif)",
            r"\bFederal Register\b",
        ],
    },
]


def _rule_check(draft: str) -> list:
    results = []
    for criterion in APA_CRITERIA:
        matched = any(re.search(p, draft, re.IGNORECASE) for p in criterion["patterns"])
        results.append({
            "id": criterion["id"],
            "label": criterion["label"],
            "desc": criterion["desc"],
            "status": "pass" if matched else "fail",
            "note": "",
            "source": "rule",
        })
    return results


def compute_score(criteria: list) -> int:
    if not criteria:
        return 0
    passed = sum(1 for c in criteria if c["status"] == "pass")
    return round(passed / len(criteria) * 100)


def check_apa(draft: str) -> dict:
    """
    Rule-based APA compliance check.
    Returns {score, level, criteria}.
    """
    if not draft or len(draft.strip()) < 30:
        return {
            "score": 0,
            "level": "incomplete",
            "criteria": [
                {
                    "id": c["id"],
                    "label": c["label"],
                    "desc": c["desc"],
                    "status": "incomplete",
                    "note": "",
                    "source": "rule",
                }
                for c in APA_CRITERIA
            ],
        }

    criteria = _rule_check(draft)
    score = compute_score(criteria)
    level = "compliant" if score >= 80 else "partial" if score >= 50 else "deficient"
    return {"score": score, "level": level, "criteria": criteria}
