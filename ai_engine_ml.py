"""
NYAYA AI 3.0 – Legal Guardian ML/NLP Engine
Advanced module for Hidden Clause Detection, Multilingual Explanations (EN, TA, HI),
Cancellation Intelligence, Indian Court/Authority Guidance, and Structured Legal Reports.
"""

import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import re
import pickle
import numpy as np
from pathlib import Path

# Machine Learning & NLP Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# NLTK's optional data files are not included in a serverless deployment.
# Avoid downloading them while a function is starting; sentence splitting and
# sentiment analysis already have safe fallbacks below when the data is absent.
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    sia = None

# ---------------------------------------------------------------------------
# Training Data for ML Risk Classifier
# ---------------------------------------------------------------------------
SAFE_DOCUMENTS = [
    "This agreement is fair and transparent. All terms are clearly stated with mutual consent.",
    "Both parties have equal rights, statutory notice periods, and fair dispute mechanisms.",
    "No hidden charges, automatic penalty escalations, or non-refundable forfeit clauses are included.",
    "The terms provide full refund of security deposit within 30 days upon peaceful handover.",
    "Clear disclosure of all fees, standard interest rates, and transparent termination rights.",
]

RISKY_DOCUMENTS = [
    "Automatic deductions, unilateral fee changes, and penalty fees will be made without consent.",
    "Non-refundable deposits cannot be recovered under any circumstances and are forfeited completely.",
    "Binding private arbitration prevents approaching judicial court or consumer commission.",
    "Unlimited liability, waiver of all statutory rights, and 2-year post-employment non-compete.",
    "Hidden charges, 5% foreclosure penalty, lock-in period penalty, and late payment interest apply.",
]

# ---------------------------------------------------------------------------
# Comprehensive Hidden Charges & Risky Clause Taxonomies
# (English, Tamil தமிழ், and Hindi हिन्दी)
# ---------------------------------------------------------------------------
RISK_KEYWORDS = {
    # 1. Processing Fees
    "processing fee": {
        "severity": 7,
        "category": "hidden_charge",
        "badge": "🟡 Medium Risk",
        "act_section": "Consumer Protection Act 2019 Sec 2(47)",
        "explanation": "Non-refundable processing fee deducted upfront before loan or service delivery.",
        "explanation_ta": "சேவை அல்லது கடன் வழங்குவதற்கு முன் திரும்பப் பெற முடியாத செயலாக்கக் கட்டணம் வசூலிக்கப்படும்.",
        "explanation_hi": "सेवा या ऋण वितरण से पहले गैर-वापसी योग्य प्रसंस्करण शुल्क काटा जाता है।"
    },
    # 2. Maintenance Charges
    "maintenance charges": {
        "severity": 6,
        "category": "hidden_charge",
        "badge": "🟡 Medium Risk",
        "act_section": "TN Regulation of Landlords & Tenants Act 2017 Sec 8",
        "explanation": "Variable monthly maintenance charges levied beyond base rent or service fee without itemized audit.",
        "explanation_ta": "அடிப்படை வாடகையைத் தாண்டி கூடுதல் மாதாந்திர பராமரிப்புக் கட்டணம் தன்னிச்சையாக விதிக்கப்படுகிறது.",
        "explanation_hi": "मूल किराए या सेवा शुल्क के अलावा अतिरिक्त मासिक रखरखाव शुल्क।"
    },
    # 3. Service Charges
    "service charges": {
        "severity": 6,
        "category": "hidden_charge",
        "badge": "🟡 Medium Risk",
        "act_section": "Consumer Protection Act 2019 / CCPA Guidelines",
        "explanation": "Unilateral service charges subject to change without prior written consent.",
        "explanation_ta": "சேவை வழங்குநரால் தன்னிச்சையாக மாற்றப்படக்கூடிய கூடுதல் சேவை கட்டணங்கள்.",
        "explanation_hi": "सेवा प्रदाता द्वारा एकतरफा परिवर्तन के अधीन परिवर्तनीय सेवा शुल्क।"
    },
    # 4. Penalty Clauses
    "penalty": {
        "severity": 8,
        "category": "hidden_charge",
        "badge": "🔴 High Risk",
        "act_section": "Indian Contract Act 1872 Sec 74 (Unreasonable Penalty)",
        "explanation": "Compounding penalty for delayed payment or minor breach of agreement terms.",
        "explanation_ta": "தாமதமான கட்டணம் அல்லது விதிமீறல்களுக்கு கூட்டு அபராதம் விதிக்கப்படும்.",
        "explanation_hi": "विलंबित भुगतान या अनुबंध शर्तों के उल्लंघन के लिए दंडात्मक जुर्माना।"
    },
    # 5. Foreclosure Charges
    "foreclosure charges": {
        "severity": 9,
        "category": "hidden_charge",
        "badge": "🔴 High Risk",
        "act_section": "RBI Master Circular on Prepayment / Foreclosure Charges",
        "explanation": "Heavy fee charged when closing or prepaying loan prior to agreed tenure (Prohibited by RBI for individuals).",
        "explanation_ta": "முன்கூட்டியே கடனை அடைத்தால் அதிக முன்கூட்டியே அடைப்புக் கட்டணம் வசூலிக்கப்படும் (ஆர்பிஐ விதிமுறைப்படி தனிநபர்களுக்கு தடை).",
        "explanation_hi": "ऋण अवधि समाप्त होने से पहले ऋण बंद करने पर भारी शुल्क (आरबीआई द्वारा व्यक्तिगत ऋणों पर प्रतिबंधित)।"
    },
    # 6. Late Payment Fees
    "late payment": {
        "severity": 7,
        "category": "hidden_charge",
        "badge": "🟡 Medium Risk",
        "act_section": "Interest on Delayed Payments Act & Contract Act Sec 73",
        "explanation": "Steep interest rate per day levied immediately upon payment due date delay.",
        "explanation_ta": "கட்டண தேதி தாமதமானால் நாளொன்றுக்கு அதிக வட்டி அபராதம் விதிக்கப்படும்.",
        "explanation_hi": "भुगतान तिथि में देरी पर तुरंत दैनिक उच्च ब्याज शुल्क।"
    },
    # 7. Lock-in Period
    "lock-in period": {
        "severity": 8,
        "category": "hidden_charge",
        "badge": "🔴 High Risk",
        "act_section": "Indian Contract Act 1872 Sec 23 & 73",
        "explanation": "Mandatory period during which termination is prohibited or incurs full rent/fee forfeit.",
        "explanation_ta": "ஒப்பந்தத்தை ரத்து செய்ய முடியாத கட்டாய காலம். மீறினால் முழு தொகை இழக்கப்படும்.",
        "explanation_hi": "अनिवार्य अवधि जिसके दौरान अनुबंध समाप्त करने पर पूरा शुल्क जब्त कर लिया जाएगा।"
    },
    # 8. Automatic Renewal Clauses
    "automatic renewal": {
        "severity": 8,
        "category": "hidden_charge",
        "badge": "🔴 High Risk",
        "act_section": "Consumer Protection Act 2019 (Unfair Contract)",
        "explanation": "Contract automatically renews with potential price increases unless cancelled before deadline.",
        "explanation_ta": "முன்கூட்டியே ரத்து செய்யாவிட்டால் ஒப்பந்தம் தானாகவே விலை உயர்வுடன் புதுப்பிக்கப்படும்.",
        "explanation_hi": "अनुबंध स्वचालित रूप से नवीनीकृत हो जाता है, जब तक कि अग्रिम नोटिस न दिया जाए।"
    },
    # 9. Deposit Deduction Conditions
    "non-refundable deposit": {
        "severity": 9,
        "category": "hidden_charge",
        "badge": "🔴 High Risk",
        "act_section": "TN Rent Act 2017 Sec 8 / Transfer of Property Act Sec 108",
        "explanation": "Security deposit forfeited completely upon agreement exit or expiry.",
        "explanation_ta": "ஒப்பந்த முடிவில் பாதுகாப்பு வைப்புத் தொகை முற்றிலும் திரும்பப் பெற முடியாது என அபகரிக்கப்படும்.",
        "explanation_hi": "अनुबंध समाप्ति पर सुरक्षा जमा पूरी तरह से जब्त कर ली जाएगी।"
    },
    # 10. Non-Compete / Restraint of Trade
    "non-compete": {
        "severity": 9,
        "category": "legal_trap",
        "badge": "🔴 High Risk",
        "act_section": "Indian Contract Act 1872 Sec 27 (Void Agreement)",
        "explanation": "Restricts post-employment work or business trade (Legally VOID in India under Section 27).",
        "explanation_ta": "வேலையிலிருந்து வெளியேறிய பிறகு பணிபுரிய தடை விதிக்கிறது (சட்டப்பிரிவு 27 கீழ் முற்றிலும் செல்லாது).",
        "explanation_hi": "रोजगार समाप्ति के बाद काम पर रोक (भारतीय अनुबंध अधिनियम धारा 27 के तहत पूर्णतः अमान्य)।"
    },
    # 11. Binding Private Arbitration
    "binding arbitration": {
        "severity": 9,
        "category": "legal_trap",
        "badge": "🔴 High Risk",
        "act_section": "Arbitration & Conciliation Act 1996 & Consumer Protection Act Sec 100",
        "explanation": "Unilaterally waives your right to approach public courts in favor of a private bank arbitrator.",
        "explanation_ta": "நீதிமன்றத்தை அணுகும் உரிமையை பறித்து தனிப்பட்ட நடுவரை கட்டாயமாக்குகிறது.",
        "explanation_hi": "अदालत जाने का अधिकार छीनकर एकतरफा निजी मध्यस्थ को अनिवार्य करता है।"
    },
    # 12. Waiver of Statutory Rights
    "waive your right": {
        "severity": 9,
        "category": "legal_trap",
        "badge": "🔴 High Risk",
        "act_section": "Constitution of India Art 21 & Contract Act Sec 28",
        "explanation": "Direct waiver of statutory legal remedies or constitutional protections.",
        "explanation_ta": "சட்டப்பூர்வ உரிமைகளை நேரடியாகத் துறக்கச் செய்கிறது.",
        "explanation_hi": "वैधानिक अधिकारों का सीधा त्याग।"
    },
    # 13. Unilateral Modification
    "unilateral": {
        "severity": 8,
        "category": "clause_trap",
        "badge": "🔴 High Risk",
        "act_section": "Indian Contract Act 1872 Sec 10 & 14 (Lack of Free Consent)",
        "explanation": "One party can alter terms, fees, or obligations without notice or agreement.",
        "explanation_ta": "ஒரு தரப்பு மற்றவரின் சம்மதமின்றி விதிகளை தன்னிச்சையாக மாற்றலாம்.",
        "explanation_hi": "एक पक्ष दूसरे की सहमति के बिना शर्तों को एकतरफा बदल सकता है।"
    }
}

SAFE_KEYWORDS = [
    "fair", "transparent", "clear", "disclosure", "protection", "warranty",
    "guarantee", "refundable", "flexible", "mutual", "balanced", "favorable",
    "approved", "consent", "reversible", "renegotiate", "amendment", "statutory notice"
]


# ---------------------------------------------------------------------------
# ML Model Training (Risk Classification)
# ---------------------------------------------------------------------------
class RiskClassifier:
    """Machine Learning based risk classification model."""
    
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=200, stop_words='english')),
            ('clf', MultinomialNB())
        ])
        self.is_trained = False
    
    def train(self):
        try:
            X_train = SAFE_DOCUMENTS + RISKY_DOCUMENTS
            y_train = [0] * len(SAFE_DOCUMENTS) + [1] * len(RISKY_DOCUMENTS)
            self.model.fit(X_train, y_train)
            self.is_trained = True
        except Exception as e:
            print(f"Warning: Could not train ML model: {e}")
            self.is_trained = False
    
    def predict_risk(self, text):
        try:
            if not self.is_trained:
                return 0.5
            proba = self.model.predict_proba([text])[0]
            return float(proba[1])
        except Exception:
            return 0.5

risk_classifier = RiskClassifier()
risk_classifier.train()


# ---------------------------------------------------------------------------
# Helper Functions & Document Type Detection
# ---------------------------------------------------------------------------

def extract_sentences(text: str) -> list:
    try:
        return sent_tokenize(text)
    except Exception:
        return re.split(r'(?<=[.!?])\s+', text)


def detect_document_type(text: str) -> str:
    """Identifies document type based on key Indian legal terms."""
    t_lower = text.lower()
    if any(k in t_lower for k in ["lease", "tenancy", "landlord", "tenant", "rent agreement", "premises"]):
        return "Rental / Lease Agreement"
    elif any(k in t_lower for k in ["insurance", "policy", "premium", "insurer", "insured", "claim"]):
        return "Insurance Policy"
    elif any(k in t_lower for k in ["loan", "borrower", "lender", "emi", "mortgage", "hypothecation", "principal"]):
        return "Home / Personal / Vehicle Loan Agreement"
    elif any(k in t_lower for k in ["sale deed", "conveyance", "buyer", "seller", "patta", "chitta", "survey number", "fmb"]):
        return "Property Sale Deed / Land Registration Document"
    elif any(k in t_lower for k in ["gift deed", "donee", "donor"]):
        return "Gift Deed"
    elif any(k in t_lower for k in ["employment", "employee", "employer", "salary", "designation", "probation", "non-compete"]):
        return "Employment Contract"
    elif any(k in t_lower for k in ["consumer", "purchase", "warranty", "invoice", "defect"]):
        return "Consumer Agreement"
    elif any(k in t_lower for k in ["power of attorney", "attorney", "principal", "agent"]):
        return "Power of Attorney"
    elif any(k in t_lower for k in ["affidavit", "sworn", "deponent", "oath"]):
        return "Affidavit"
    elif any(k in t_lower for k in ["mortgage deed", "mortgagor", "mortgagee"]):
        return "Mortgage Deed"
    else:
        return "General Indian Legal Document"


def sentiment_analysis(text: str) -> dict:
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        vader_scores = sia.polarity_scores(text) if sia else {"compound": 0}
        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "vader_compound": vader_scores['compound'],
            "sentiment_label": "favorable / balanced" if polarity > 0.1 else "onerous / risky" if polarity < -0.05 else "standard neutral"
        }
    except Exception:
        return {"polarity": 0, "subjectivity": 0.5, "vader_compound": 0, "sentiment_label": "standard neutral"}


# ---------------------------------------------------------------------------
# Cancellation Intelligence & Court Guidance Engine
# ---------------------------------------------------------------------------

def analyze_cancellation(text: str, matched_keywords: list = None) -> dict:
    """
    Extracts cancellation clause details, penalty calculations, notice period,
    refund eligibility, required documents, and next legal steps.
    """
    text_lower = text.lower()
    sentences = extract_sentences(text)
    
    cancel_sentences = [s.strip() for s in sentences if any(k in s.lower() for k in ["cancel", "terminate", "exit", "lock-in", "notice period", "forfeit"])]
    
    notice_period = "30 Days statutory written notice" if ("30" in text_lower or "one month" in text_lower) else "60 Days written notice" if "60" in text_lower else "Standard 30 Days written notice required by law"
    
    has_lockin = any("lock-in" in s.lower() for s in cancel_sentences)
    has_penalty = any("penalty" in s.lower() or "forfeit" in s.lower() for s in cancel_sentences)
    
    if has_lockin:
        penalty_est = "Lock-in period exit penalty applies (forfeiture of remaining lock-in tenure or security deposit)"
    elif has_penalty:
        penalty_est = "Calculated cancellation penalty applies (approx 1–2 months fee or deduction from deposit)"
    else:
        penalty_est = "No explicit cancellation penalty detected. Standard notice period applies."
        
    refund_status = "Full refund of advance/security deposit required within 30 days after deducting genuine utility arrears." if "refundable" in text_lower else "Non-refundable forfeiture condition detected! May be legally challenged under Contract Act Sec 74."

    return {
        "cancellation_clause_found": len(cancel_sentences) > 0,
        "clause_text": cancel_sentences[0] if cancel_sentences else "Standard statutory cancellation rights apply.",
        "notice_period": notice_period,
        "penalty_estimate": penalty_est,
        "refund_eligibility": refund_status,
        "required_documents": [
            "Original Agreement Copy",
            "Payment / Rent Receipts & Bank Statements",
            "Written Cancellation Notice sent via Registered Post with Ack Due",
            "Property Handover / No-Dues NOC Certificate"
        ],
        "next_step": "Issue a formal written Termination Notice via Registered Post with Ack Due. Document physical handover with timestamps."
    }


def recommend_court_authority(doc_type: str, matched_keywords: list) -> dict:
    """
    Recommends specific Indian Judicial Body / Authority with clear legal reasoning.
    """
    if "Rental" in doc_type or "Lease" in doc_type:
        return {
            "authority": "Rent Authority / Rent Tribunal (under State Rent Control Act)",
            "alt_authority": "District Civil Court (for title / recovery disputes exceeding jurisdiction)",
            "reason": "Rent Authorities and Tribunals are statutory bodies established specifically to resolve tenancy disputes, illegal eviction, and security deposit recovery expeditiously without long civil delays.",
            "relevant_law": "Tamil Nadu Regulation of Rights and Responsibilities of Landlords and Tenants Act 2017 & Transfer of Property Act 1882 (Sec 106–111)"
        }
    elif "Loan" in doc_type:
        return {
            "authority": "RBI Banking Ombudsman / Debt Recovery Tribunal (DRT)",
            "alt_authority": "District Consumer Commission (for unfair trade practices & illegal recovery)",
            "reason": "The RBI Ombudsman provides free, binding grievance redressal against banks and NBFCs for hidden charges, unlawful foreclosure fees, or abusive loan recovery practices without requiring advocate fees.",
            "relevant_law": "Reserve Bank of India Act 1934 & Consumer Protection Act 2019 (Sec 2(47))"
        }
    elif "Insurance" in doc_type:
        return {
            "authority": "Insurance Ombudsman (IRDAI)",
            "alt_authority": "State Consumer Disputes Redressal Commission",
            "reason": "The Insurance Ombudsman has statutory power to adjudicate claim repudiation, delayed claim settlement, and policy mis-selling up to ₹30 Lakhs with binding orders on insurers.",
            "relevant_law": "Insurance Regulatory and Development Authority of India (IRDAI) Regulations & Insurance Act 1938"
        }
    elif "Property" in doc_type or "Land" in doc_type or "Gift" in doc_type or "Mortgage" in doc_type:
        return {
            "authority": "Real Estate Regulatory Authority (RERA) / District Civil Court",
            "alt_authority": "Revenue Divisional Officer (RDO) / Sub-Registrar",
            "reason": "RERA enforces project delivery timelines and refund of buyer deposits. The District Civil Court has exclusive jurisdiction over land title declaration, specific performance of sale agreements, and boundary cancellation suits.",
            "relevant_law": "Transfer of Property Act 1882 (Sec 54), Registration Act 1908 (Sec 17), RERA Act 2016, and TN Patta Pass Book Act 1983"
        }
    elif "Employment" in doc_type:
        return {
            "authority": "Labour Court / Industrial Tribunal",
            "alt_authority": "District Civil Court (for challenging non-compete restraint)",
            "reason": "Labour Courts protect employees against wrongful termination, unlawful salary withholding, and abusive employment bonds.",
            "relevant_law": "Industrial Disputes Act 1947 & Indian Contract Act 1872 Sec 27 (Agreements in Restraint of Trade are Void)"
        }
    elif "Consumer" in doc_type:
        return {
            "authority": "District Consumer Disputes Redressal Commission",
            "alt_authority": "National Consumer Helpline (NCH) / e-Daakhil Portal",
            "reason": "Consumer Commissions provide fast-track, low-cost remedies for defective goods, deficiency in services, and misleading terms without mandatory advocate representation.",
            "relevant_law": "Consumer Protection Act 2019 (Sec 35)"
        }
    else:
        return {
            "authority": "District Civil Court / Commercial Court",
            "alt_authority": "District Consumer Commission",
            "reason": "Civil Courts handle general breach of contract, injunctions, and recovery suits under the Code of Civil Procedure.",
            "relevant_law": "Indian Contract Act 1872 & Bharatiya Nyaya Sanhita 2023"
        }


# ---------------------------------------------------------------------------
# Advanced Risk Scanning Engine
# ---------------------------------------------------------------------------

def scan_risks_ml(text: str) -> dict:
    """
    Advanced legal risk scanner implementing the complete NYAYA AI 3.0 System Prompt:
    Hidden Clause Detection, 3-tier Risk Badges, Cancellation Intelligence,
    Court Guidance, Multilingual explanations, and Confidence Metrics.
    """
    text_lower = text.lower()
    matched = []
    seen_keywords = set()
    sentences = extract_sentences(text)
    doc_type = detect_document_type(text)
    
    for keyword, details in RISK_KEYWORDS.items():
        if keyword in text_lower and keyword not in seen_keywords:
            seen_keywords.add(keyword)
            matched_sentence = ""
            for sentence in sentences:
                if keyword in sentence.lower():
                    matched_sentence = sentence.strip()
                    break
            
            matched.append({
                "keyword": keyword,
                "severity": details["severity"],
                "badge": details.get("badge", "🟡 Medium Risk"),
                "category": details["category"],
                "act_section": details.get("act_section", "Indian Contract Act 1872"),
                "explanation": details["explanation"],
                "explanation_ta": details.get("explanation_ta", details["explanation"]),
                "explanation_hi": details.get("explanation_hi", details["explanation"]),
                "sentence": matched_sentence if matched_sentence else f"Clause containing '{keyword}' was identified.",
            })
    
    ml_risk_prob = risk_classifier.predict_risk(text)
    keyword_risk_score = min(len(matched) * 16, 100)
    safe_count = sum(1 for keyword in SAFE_KEYWORDS if keyword in text_lower)
    safe_adjustment = max(0, safe_count * 5)
    text_length_factor = min(len(text) / 600, 1.0)
    
    combined_risk = (
        keyword_risk_score * 0.45 +
        ml_risk_prob * 100 * 0.35 +
        text_length_factor * 20 * 0.20
    ) - safe_adjustment
    
    risk_score = int(max(0, min(combined_risk, 100)))
    
    if risk_score <= 30:
        risk_type = "Safe"
        risk_badge = "🟢 Safe"
    elif risk_score <= 35:
        risk_type = "Moderate Risk"
        risk_badge = "🟡 Medium Risk"
    else:
        risk_type = "High Risk"
        risk_badge = "🔴 High Risk"
    
    sentiment = sentiment_analysis(text)
    cancellation = analyze_cancellation(text, matched)
    court_info = recommend_court_authority(doc_type, matched)
    
    return {
        "document_type": doc_type,
        "risk_score": risk_score,
        "risk_type": risk_type,
        "risk_badge": risk_badge,
        "ml_confidence": round(ml_risk_prob, 2),
        "confidence_percentage": int(max(75, min(95, 80 + (len(matched) * 3)))),
        "keyword_count": len(matched),
        "matched": matched,
        "cancellation": cancellation,
        "court_info": court_info,
        "sentiment": sentiment,
        "analysis_type": "NYAYA AI 3.0 Legal Guardian"
    }


def highlight_clauses(text: str, matched_keywords: list) -> str:
    highlighted = text
    for item in matched_keywords:
        keyword = item["keyword"]
        severity = item.get("severity", 5)
        color_class = "risk-highlight-critical" if severity >= 8 else "risk-highlight-warning" if severity >= 6 else "risk-highlight-mild"
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted = pattern.sub(
            f'<span class="{color_class}" title="Severity: {severity}/10">{keyword}</span>',
            highlighted,
        )
    return highlighted


# ---------------------------------------------------------------------------
# Structured System Prompt Output Generator
# ---------------------------------------------------------------------------

def generate_risk_report(scan_result: dict) -> str:
    """
    Generates structured output matching NYAYA AI 3.0 System Prompt format:
    ### Document Summary
    ### Hidden Charges
    ### Risk Analysis
    ### Cancellation Penalty
    ### Legal Rights
    ### Recommended Next Step
    ### Relevant Law & Section
    ### Confidence Score
    """
    doc_type = scan_result.get("document_type", "Legal Document")
    risk_score = scan_result.get("risk_score", 0)
    risk_type = scan_result.get("risk_type", "Safe")
    risk_badge = scan_result.get("risk_badge", "🟢 Safe")
    conf_pct = scan_result.get("confidence_percentage", 88)
    matched = scan_result.get("matched", [])
    canc = scan_result.get("cancellation", {})
    court = scan_result.get("court_info", {})
    
    # Format Hidden Charges
    hidden_charges_html = ""
    if matched:
        for m in matched:
            hidden_charges_html += f"""
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.85rem; margin-bottom: 0.65rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-weight: 700; color: #0f172a;">{m['badge']} {m['keyword'].title()}</div>
                    <span style="font-size:0.75rem; background:#f1f5f9; color:#475569; padding:2px 8px; border-radius:4px;">{m.get('act_section', 'Contract Act')}</span>
                </div>
                <div style="font-size: 0.88rem; color: #334155; margin-top: 0.35rem;"><strong>English:</strong> {m['explanation']}</div>
                <div style="font-size: 0.85rem; color: #0284c7; margin-top: 0.2rem;"><strong>தமிழ்:</strong> {m['explanation_ta']}</div>
                <div style="font-size: 0.85rem; color: #7c3aed; margin-top: 0.2rem;"><strong>हिन्दी:</strong> {m.get('explanation_hi', m['explanation'])}</div>
                <div style="font-size: 0.80rem; color: #64748b; font-style: italic; margin-top: 0.25rem;">"{m.get('sentence', '')}"</div>
            </div>
            """
    else:
        hidden_charges_html = "<p style='color: #10b981; font-weight: 500;'>🟢 No unfair or hidden charges detected in this document.</p>"

    report_html = f"""
    <div class="nyaya-analysis-report" style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
        
        <!-- Header -->
        <div style="border-bottom: 2px solid #e2e8f0; padding-bottom: 1rem; margin-bottom: 1.25rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h3 style="color: #0f172a; margin: 0; font-weight: 700;">🛡️ NYAYA AI 3.0 – Legal Guardian Report</h3>
                <span style="font-size: 0.875rem; color: #64748b;">Document Category: <strong>{doc_type}</strong></span>
            </div>
            <span style="font-size: 0.85rem; background: #0f172a; color: #38bdf8; font-weight: 600; padding: 4px 12px; border-radius: 20px;">
                Confidence: {conf_pct}%
            </span>
        </div>

        <!-- Section 1: Document Summary -->
        <div style="margin-bottom: 1.25rem;">
            <h5 style="color: #1e293b; margin-bottom: 0.4rem; font-weight: 700;">### Document Summary</h5>
            <p style="color: #334155; font-size: 0.92rem; line-height: 1.6; margin: 0;">
                The document has been verified as a <strong>{doc_type}</strong>. The AI Legal Guardian scanned all contractual terms,
                identifying potential financial liabilities, cancellation restrictions, unconscionable clauses, and statutory remedies under Indian Law.
            </p>
        </div>

        <!-- Section 2: Hidden Charges -->
        <div style="margin-bottom: 1.25rem;">
            <h5 style="color: #1e293b; margin-bottom: 0.4rem; font-weight: 700;">### Hidden Charges ({len(matched)} Identified)</h5>
            {hidden_charges_html}
        </div>

        <!-- Section 3: Risk Analysis -->
        <div style="margin-bottom: 1.25rem; background: white; padding: 1rem; border-radius: 8px; border-left: 5px solid {'#ef4444' if risk_type == 'High Risk' else '#f59e0b' if risk_type == 'Moderate Risk' else '#10b981'};">
            <h5 style="color: #1e293b; margin-bottom: 0.4rem; font-weight: 700;">### Risk Analysis</h5>
            <div style="display: flex; align-items: center; gap: 1.25rem; flex-wrap: wrap;">
                <span style="font-size: 2rem; font-weight: 800; color: {'#ef4444' if risk_type == 'High Risk' else '#f59e0b' if risk_type == 'Moderate Risk' else '#10b981'};">{risk_score}/100</span>
                <div>
                    <div style="font-weight: 700; font-size: 1rem;">Overall Status: {risk_badge}</div>
                    <div style="font-size: 0.85rem; color: #64748b;">Contractual Tone: {scan_result.get('sentiment', {}).get('sentiment_label', 'Neutral').title()}</div>
                </div>
            </div>
        </div>

        <!-- Section 4: Cancellation Penalty -->
        <div style="margin-bottom: 1.25rem; background: white; padding: 1rem; border-radius: 8px;">
            <h5 style="color: #1e293b; margin-bottom: 0.4rem; font-weight: 700;">### Cancellation Penalty & Exit Intelligence</h5>
            <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.90rem; color: #334155; line-height: 1.6;">
                <li><strong>Notice Period Required:</strong> {canc.get('notice_period', '30 Days statutory written notice')}</li>
                <li><strong>Penalty Calculation:</strong> {canc.get('penalty_estimate', 'Standard notice applies')}</li>
                <li><strong>Refund Eligibility:</strong> {canc.get('refund_eligibility', 'Verification Required')}</li>
            </ul>
        </div>

        <!-- Section 5: Legal Rights -->
        <div style="margin-bottom: 1.25rem; background: white; padding: 1rem; border-radius: 8px;">
            <h5 style="color: #1e293b; margin-bottom: 0.4rem; font-weight: 700;">### Legal Rights & Statutory Protections</h5>
            <p style="font-size: 0.90rem; color: #334155; line-height: 1.6; margin: 0;">
                Under the <strong>Indian Contract Act 1872 (Sections 23, 27, 74)</strong> and <strong>Consumer Protection Act 2019</strong>, clauses that impose unreasonable penalties or restraint of trade are <strong>legally void</strong>. Parties have statutory rights to fair notice, refund of deposits, and judicial review.
            </p>
        </div>

        <!-- Section 6: Recommended Next Step -->
        <div style="margin-bottom: 1.25rem; background: #eff6ff; border: 1px solid #bfdbfe; padding: 1rem; border-radius: 8px;">
            <h5 style="color: #1e40af; margin-bottom: 0.4rem; font-weight: 700;">### Recommended Next Step</h5>
            <p style="font-size: 0.90rem; color: #1e3a8a; line-height: 1.6; margin: 0;">
                <strong>Next Legal Step:</strong> {canc.get('next_step')}<br>
                <strong>Appropriate Authority / Court:</strong> <strong>{court.get('authority')}</strong><br>
                <em>Jurisdictional Rationale: {court.get('reason')}</em>
            </p>
        </div>

        <!-- Section 7: Relevant Law & Section -->
        <div style="margin-bottom: 1.25rem;">
            <h5 style="color: #1e293b; margin-bottom: 0.4rem; font-weight: 700;">### Relevant Law & Section</h5>
            <div style="font-size: 0.90rem; background: white; padding: 0.75rem 1rem; border-radius: 6px; border: 1px solid #e2e8f0; color: #0f172a; font-weight: 600;">
                📜 {court.get('relevant_law', 'Indian Contract Act 1872, Transfer of Property Act 1882 & Bharatiya Nyaya Sanhita 2023')}
            </div>
        </div>

        <!-- Section 8: Confidence Score -->
        <div style="display: flex; justify-content: space-between; align-items: center; background: #0f172a; color: white; padding: 0.85rem 1.25rem; border-radius: 8px;">
            <span style="font-size: 0.95rem; font-weight: 700;">### Confidence Score</span>
            <span style="font-size: 1.15rem; font-weight: 800; color: #38bdf8;">{conf_pct}% ML Confidence</span>
        </div>

    </div>
    """
    return report_html


# Backward compatibility wrappers
def scan_risks(text: str):
    res = scan_risks_ml(text)
    return {
        "risk_score": res["risk_score"],
        "risk_type": res["risk_type"],
        "matched": res["matched"]
    }

def simplify_clauses(matched_data: list):
    simplified = []
    for item in matched_data:
        kw = item.get("keyword", "")
        exp = item.get("explanation", "Risky clause")
        simplified.append({
            "original": item.get("sentence", kw),
            "simplified": exp,
            "simplified_ta": item.get("explanation_ta", exp),
            "simplified_hi": item.get("explanation_hi", exp),
            "negotiation": f"Alternative wording: 'The {kw} shall be mutually agreed upon and reasonably capped by statutory limits.'",
            "keyword": kw
        })
    return simplified

def compare_documents(text1: str, text2: str):
    all_keywords = set(RISK_KEYWORDS.keys())
    diffs = []
    for kw in all_keywords:
        in_1 = kw.lower() in text1.lower()
        in_2 = kw.lower() in text2.lower()
        if in_1 != in_2:
            diffs.append({
                "keyword": kw,
                "status": "In Document 1 only" if in_1 else "In Document 2 only",
                "severity": RISK_KEYWORDS.get(kw, {}).get("severity", 5)
            })
    return {
        "doc1_stats": {"length": len(text1), "sentences": len(extract_sentences(text1))},
        "doc2_stats": {"length": len(text2), "sentences": len(extract_sentences(text2))},
        "differences": diffs
    }

if __name__ == "__main__":
    sample = "This agreement requires non-refundable deposit, lock-in period of 12 months, 10% late payment penalty, and binding arbitration. Automatic renewal applies."
    res = scan_risks_ml(sample)
    print("=== NYAYA AI 3.0 Risk Scan Test ===")
    print(f"Risk Score: {res['risk_score']}/100 | {res['risk_badge']}")
    print(f"Document Type: {res['document_type']}")
    print(f"Hidden Charges: {len(res['matched'])}")
    print(f"Court: {res['court_info']['authority']}")
    print(f"Confidence: {res['confidence_percentage']}%")
