"""
NYAYA AI 3.0 – AI Legal Risk Scanner Engine with ML/NLP
Integrates advanced machine learning (scikit-learn, NLTK, TextBlob)
with rule-based NLP for enhanced risk detection, hidden clause analysis, and structured reporting.
"""

import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import re

# ============ Try to Import ML/NLP Module ============
ML_AVAILABLE = False
ml_module = None
try:
    import ai_engine_ml as ml_module
    ML_AVAILABLE = True
    print("✅ ML/NLP module loaded successfully")
except ImportError as e:
    print(f"⚠️ ML module not available ({e}). Using legacy mode.")

# ---------------------------------------------------------------------------
# Fallback Legacy Risk keywords
# ---------------------------------------------------------------------------
RISK_KEYWORDS = {
    "penalty": "You may have to pay extra money as a punishment for breaking a rule.",
    "hidden charges": "There may be extra costs that are not clearly mentioned upfront.",
    "termination fee": "You may have to pay a fee if you end the agreement early.",
    "liability": "You could be held legally responsible for damages or losses.",
    "automatic rent increase": "Your rent can go up automatically without your approval.",
    "non-refundable deposit": "The deposit you pay will NOT be returned to you.",
    "non refundable deposit": "The deposit you pay will NOT be returned to you.",
    "increase rent": "Your rent can be increased, possibly without prior notice.",
    "additional charges": "There may be extra costs beyond what is stated.",
    "waive your right": "You might be giving up an important legal right.",
    "binding arbitration": "You may not be able to take the matter to court.",
    "indemnify": "You may be required to compensate the other party for their losses.",
    "forfeit": "You could lose money or rights under certain conditions.",
}

KEYWORD_WEIGHT = 100 / 6


def scan_risks_legacy(text: str):
    """Legacy keyword-based risk detection (fallback)."""
    text_lower = text.lower()
    matched = []
    seen_keywords = set()
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for keyword, explanation in RISK_KEYWORDS.items():
        if keyword in text_lower and keyword not in seen_keywords:
            seen_keywords.add(keyword)
            matched_sentence = ""
            for sentence in sentences:
                if keyword in sentence.lower():
                    matched_sentence = sentence.strip()
                    break
            matched.append({
                "keyword": keyword,
                "badge": "🟡 Medium Risk",
                "explanation": explanation,
                "explanation_ta": explanation,
                "explanation_hi": explanation,
                "sentence": matched_sentence,
            })

    hit_count = len(matched)
    risk_score = min(int(hit_count * KEYWORD_WEIGHT), 100)

    if risk_score <= 30:
        risk_type = "Safe"
        risk_badge = "🟢 Safe"
    elif risk_score <= 35:
        risk_type = "Moderate Risk"
        risk_badge = "🟡 Medium Risk"
    else:
        risk_type = "High Risk"
        risk_badge = "🔴 High Risk"

    return {
        "document_type": "General Legal Agreement",
        "risk_score": risk_score,
        "confidence_score": 100,
        "risk_type": risk_type,
        "risk_badge": risk_badge,
        "matched": matched,
        "confidence_percentage": 100,
        "cancellation": {
            "notice_period": "30 Days written notice",
            "penalty_estimate": "Subject to agreement terms",
            "refund_eligibility": "Deposit refundable less genuine deductions",
            "next_step": "Send written notice via Registered Post."
        },
        "court_info": {
            "authority": "District Civil Court / Consumer Commission",
            "reason": "Adjudicates breach of contract and recovery disputes.",
            "relevant_law": "Indian Contract Act 1872"
        }
    }


def highlight_clauses_legacy(text: str, matched_keywords: list) -> str:
    highlighted = text
    for item in matched_keywords:
        keyword = item["keyword"]
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted = pattern.sub(
            f'<span class="risk-highlight">{keyword}</span>',
            highlighted,
        )
    return highlighted


def simplify_clauses_legacy(matched: list) -> list:
    simplified = []
    for item in matched:
        kw = item.get("keyword", "clause")
        simplified.append({
            "original": item.get("sentence", kw),
            "simplified": item.get("explanation", ""),
            "simplified_ta": item.get("explanation_ta", item.get("explanation", "")),
            "simplified_hi": item.get("explanation_hi", item.get("explanation", "")),
            "negotiation": f"Alternative wording: 'The {kw} shall be mutually agreed upon and reasonably capped by statutory limits.'"
        })
    return simplified


# ============ Smart Routing (ML if available, else Legacy) ============

def scan_risks(text: str):
    """
    Intelligent risk detection using ML/NLP if available, else legacy method.
    """
    if ML_AVAILABLE and ml_module:
        try:
            result = ml_module.scan_risks_ml(text)
            result["report_html"] = ml_module.generate_risk_report(result)
            result["ml_enabled"] = True
            return result
        except Exception as e:
            print(f"ML scan error: {e}. Falling back to legacy mode.")
            return scan_risks_legacy(text)
    else:
        return scan_risks_legacy(text)


def highlight_clauses(text: str, matched_keywords: list) -> str:
    """Route to ML or legacy highlighting."""
    if ML_AVAILABLE and ml_module:
        try:
            return ml_module.highlight_clauses(text, matched_keywords)
        except Exception:
            return highlight_clauses_legacy(text, matched_keywords)
    return highlight_clauses_legacy(text, matched_keywords)


def simplify_clauses(matched: list) -> list:
    """Route to ML or legacy simplification."""
    if ML_AVAILABLE and ml_module:
        try:
            return ml_module.simplify_clauses(matched)
        except Exception:
            return simplify_clauses_legacy(matched)
    return simplify_clauses_legacy(matched)
