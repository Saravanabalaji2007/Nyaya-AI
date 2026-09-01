"""
NYAYA AI 3.0 – AI Legal Guardian Chatbot with NLP
Advanced chatbot using Multilingual NLP similarity matching, trained Legal corpora (150+ Q&As),
fuzzy matching, and semantic understanding for Indian legal guidance.
"""

import re
from difflib import SequenceMatcher

# Import the trained chatbot model
try:
    from chatbot_trainer import trained_chatbot, preprocess_text
    TRAINED_MODEL_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Could not load trained chatbot model: {e}")
    TRAINED_MODEL_AVAILABLE = False
    trained_chatbot = None

# ---------------------------------------------------------------------------
# Enhanced Conversational & Procedural Knowledge Base
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    # ---- Greetings & Identity (Matching NYAYA AI 3.0 System Prompt) ----
    {
        "patterns": ["hello", "hi", "hey", "namaste", "vanakkam", "greetings", "good morning", "good evening", "வணக்கம்", "नमस्ते"],
        "keywords": ["hello", "hi", "namaste", "vanakkam"],
        "category": "general_conversation",
        "response": (
            "👋 <strong>Hello! I'm NYAYA AI 3.0 – Your AI Legal Guardian.</strong><br><br>"
            "How can I help you today? You can ask me about:<br>"
            "• 📄 <strong>Document Hidden Clauses:</strong> (Rental, Employment, Loan, Sale Deed)<br>"
            "• 🏛️ <strong>Court & Authority Guidance:</strong> (Consumer Court, Civil Court, RERA, Rent Authority)<br>"
            "• 🔍 <strong>Tamil Nadu Land Records:</strong> (Patta, Chitta, EC, FMB, Guideline Values)<br>"
            "• ⚖️ <strong>Indian Laws:</strong> (Constitution, BNS 2023, BNSS, Contract Act, RERA)<br>"
            "• 🌐 <strong>Multilingual:</strong> English, தமிழ் (Tamil), and हिन्दी (Hindi) supported!<br><br>"
            "<em>Type your question or choose a quick prompt below.</em>"
        )
    },
    {
        "patterns": ["who are you", "what is nyaya ai", "about you", "your name", "who made you", "நீ யார்", "आप कौन हैं"],
        "keywords": ["who", "are", "you", "about", "identity"],
        "category": "general_conversation",
        "response": (
            "🛡️ <strong>I am NYAYA AI 3.0</strong>, India's advanced multilingual AI Legal Guardian.<br><br>"
            "My mission is to empower citizens by analyzing complex legal documents, identifying hidden financial risks, "
            "verifying land records with authentic Tamil Nadu government sources, and explaining statutory legal rights in plain English, Tamil, and Hindi.<br><br>"
            "<em>Always consult a verified advocate on our <a href='/lawfirms' style='color:#818cf8;font-weight:600;'>Law Firms page</a> for court representations.</em>"
        )
    },
    # ---- Signed Risky Document – Step-by-step Procedure ----
    {
        "patterns": ["signed risky document", "signed a risky", "risky document signed",
                      "signed bad contract", "signed unfair", "what to do if i signed",
                      "signed risk", "i signed a risky", "signed dangerous document",
                      "signed wrong document", "cheated document", "fraud document"],
        "keywords": ["signed", "risky", "document", "contract", "fraud", "danger"],
        "category": "risky_document_remedy",
        "response": (
            "🚨 <strong>What To Do If You Signed a Risky Legal Document</strong><br><br>"
            "Don't panic! Follow these steps carefully:<br><br>"
            "<strong>📋 Step 1: File an FIR Complaint</strong><br>"
            "• Go to your <strong>nearest police station</strong> immediately.<br>"
            "• File a <strong>First Information Report (FIR)</strong> under BNS Section 318 (Cheating) / Section 336 (Forgery) or IPC 420.<br>"
            "• Keep a <strong>free copy of the FIR</strong> for your records.<br>"
            '<button type="button" class="btn btn-sm btn-outline-primary mt-2 mb-3 px-3 py-1" style="border-radius:20px; font-weight:600;" onclick="askQuestion(\'FIR complaint template\')"><i class="bi bi-file-text me-1"></i> Get FIR Template</button><br>'
            "<strong>📄 Step 2: Gather Evidence</strong><br>"
            "• Collect the <strong>original document</strong> and signed copies.<br>"
            "• Save all <strong>communications</strong> (emails, WhatsApp messages, bank transaction receipts).<br>"
            "• Note down <strong>witnesses</strong> who were present during signing.<br><br>"
            "<strong>⚖️ Step 3: Identify the Right Court</strong><br>"
            "What type of document is it?<br>"
            "<div class='d-flex gap-2 mt-2 mb-3 flex-wrap'>"
            "<button type='button' class='btn btn-sm btn-outline-success px-3 py-1' style='border-radius:20px; font-weight:600;' onclick='askQuestion(\"My document involves GST or defective products\")'>GST / Consumer Issue</button>"
            "<button type='button' class='btn btn-sm btn-outline-warning px-3 py-1' style='border-radius:20px; font-weight:600;' onclick='askQuestion(\"My document is a normal agreement or property contract\")'>Normal Contract / Property</button>"
            "</div>"
            "<strong>🏛️ Step 4: Consult a Lawyer</strong><br>"
            "• Based on your document type, consult a specialist advocate.<br>"
            '• Visit our <a href="/lawfirms" style="color:#818cf8;font-weight:600;">Law Firms page</a> to find the right lawyer for your case.<br><br>'
            "💡 <em>Tip: Act quickly! Many legal remedies have statutory limitation periods.</em>"
        ),
    },
    # ---- Interactive Court Selection ----
    {
        "patterns": ["gst or defective products", "gst", "defective product", "consumer issue", "consumer document"],
        "keywords": ["gst", "consumer", "defective", "product", "issue"],
        "category": "consumer_court",
        "response": (
            "🏛️ <strong>Consumer Commission (Consumer Court)</strong><br><br>"
            "Since your document involves GST overcharge, defective products, or unfair trade practices, approach the <strong>Consumer Disputes Redressal Commission</strong>.<br><br>"
            "• It is fast, affordable, and requires no compulsory advocate.<br>"
            "• You can file online at <strong>edaakhil.nic.in</strong>.<br><br>"
            '<button type="button" class="btn btn-sm btn-outline-primary mt-2" style="border-radius:20px;" onclick="askQuestion(\'Find lawyer\')"><i class="bi bi-person-badge me-1"></i>Find a Consumer Protection Lawyer</button>'
        ),
    },
    {
        "patterns": ["normal agreement or property", "normal contract", "property contract", "rental agreement", "loan agreement", "service agreement"],
        "keywords": ["normal", "contract", "property", "rental", "loan", "agreement"],
        "category": "civil_court",
        "response": (
            "🏛️ <strong>District Civil Court / Rent Authority</strong><br><br>"
            "Since your document is a regular contract, lease, or land sale deed, you should file in the <strong>Civil Court</strong> or <strong>Rent Tribunal</strong>.<br><br>"
            "• Jurisdiction depends on property location or agreement execution location.<br><br>"
            '<button type="button" class="btn btn-sm btn-outline-primary mt-2" style="border-radius:20px;" onclick="askQuestion(\'Find lawyer\')"><i class="bi bi-person-badge me-1"></i>Find a Property / Civil Lawyer</button>'
        ),
    },
    # ---- FIR Complaint Template ----
    {
        "patterns": ["fir complaint", "fir template", "fir letter", "fir format",
                      "file fir", "first information report", "police complaint",
                      "fir complaint template", "how to file fir"],
        "keywords": ["fir", "complaint", "police", "first information report"],
        "category": "fir_template",
        "response": (
            "📝 <strong>FIR Complaint Letter Template (BNSS / IPC Compliant)</strong><br><br>"
            "<div style='background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.25);border-radius:10px;padding:1.2rem;margin:0.5rem 0;font-family:monospace;font-size:0.88rem;line-height:1.7;'>"
            "<strong>To,</strong><br>"
            "The Station House Officer (SHO),<br>"
            "[Police Station Name],<br>"
            "[City/District, State]<br><br>"
            "<strong>Date:</strong> [DD/MM/YYYY]<br><br>"
            "<strong>Subject:</strong> Complaint for Registration of FIR under BNS Section 318 / 336 (IPC 420/468)<br><br>"
            "<strong>Respected Sir/Madam,</strong><br><br>"
            "I, <strong>[Your Full Name]</strong>, S/o or D/o <strong>[Father's/Husband's Name]</strong>, "
            "residing at <strong>[Your Complete Address]</strong> (Phone: [Your Phone]), lodge this complaint:<br><br>"
            "1. On <strong>[Date of Signing]</strong>, the opposite party <strong>[Name of Accused/Company]</strong> induced me to sign <strong>[Document Name]</strong> under false misrepresentations.<br>"
            "2. The document contains fraudulent / unconscionable clauses: <strong>[Describe hidden charges / lock-in / forgery]</strong>.<br>"
            "3. I have suffered a financial loss / threat of ₹<strong>[Amount]</strong>.<br><br>"
            "<strong>Prayer:</strong><br>"
            "Kindly register an FIR under relevant sections of Bharatiya Nyaya Sanhita (BNS) / IPC and take lawful action.<br><br>"
            "<strong>Yours faithfully,</strong><br>"
            "[Signature & Name]"
            "</div><br>"
            "💡 <em>Tip: Under BNSS Section 173, police must issue a Zero FIR even if the incident occurred outside their territorial jurisdiction.</em>"
        ),
    }
]


# ---------------------------------------------------------------------------
# NLP Chatbot Class with Multi-layer Matching
# ---------------------------------------------------------------------------
class NPLChatbot:
    """NLP-enhanced chatbot combining exact rules, fuzzy matching, and trained legal ML model."""
    
    def __init__(self):
        self.knowledge_base = KNOWLEDGE_BASE
    
    def _fuzzy_score(self, query: str, pattern: str) -> float:
        return SequenceMatcher(None, query.lower().strip(), pattern.lower().strip()).ratio()
    
    def find_best_match(self, query: str) -> dict:
        q_clean = query.lower().strip()
        
        # 1. Exact / High-confidence Pattern Match in Knowledge Base
        for idx, kb_item in enumerate(self.knowledge_base):
            for pattern in kb_item["patterns"]:
                if pattern in q_clean or self._fuzzy_score(q_clean, pattern) > 0.82:
                    return {
                        "response": kb_item["response"],
                        "category": kb_item.get("category", "general"),
                        "confidence": 0.96,
                        "method": "rule_pattern",
                        "found": True
                    }
        
        # 2. Trained ML/NLP Model Search (150+ legal Q&As with TF-IDF cosine similarity)
        if TRAINED_MODEL_AVAILABLE and trained_chatbot and trained_chatbot.is_ready:
            try:
                res = trained_chatbot.find_answer(query, threshold=0.12)
                if res and res.get("found") and res.get("answer"):
                    return {
                        "response": res["answer"],
                        "category": res.get("category", "legal_nlp"),
                        "confidence": res.get("confidence", 0.80),
                        "method": "trained_nlp_model",
                        "matched_question": res.get("matched_question", ""),
                        "source": res.get("source", "Indian Legal Knowledgebase"),
                        "found": True
                    }
            except Exception as e:
                print(f"Trained chatbot query error: {e}")
        
        # 3. Keyword Match Fallback in Knowledge Base
        for kb_item in self.knowledge_base:
            keywords = kb_item.get("keywords", [])
            matches = sum(1 for kw in keywords if kw in q_clean)
            if matches >= 2 or (len(keywords) == 1 and matches == 1):
                return {
                    "response": kb_item["response"],
                    "category": kb_item.get("category", "keyword"),
                    "confidence": 0.75,
                    "method": "keyword_match",
                    "found": True
                }
        
        # 4. Default Helpful Legal Guardian Fallback
        return {
            "response": (
                "⚖️ <strong>NYAYA AI 3.0 – Legal Guardian</strong><br><br>"
                "I couldn't find a direct statutory match for your specific phrasing, but I can guide you on:<br>"
                "• <strong>Criminal Laws:</strong> BNS Section 318 (Cheating), Zero FIR under BNSS, Bail rules<br>"
                "• <strong>Property & Land:</strong> TN Patta Chitta verification, Guideline values, Encumbrance Certificate (EC)<br>"
                "• <strong>Contracts & Tenancy:</strong> Non-compete validity (Sec 27), Rental agreement traps, Lock-in periods<br>"
                "• <strong>Consumer Protection:</strong> edaakhil filing, Unfair trade practices, GST disputes<br><br>"
                "<em>Try asking: 'How to verify Patta Chitta in Tamil Nadu?', 'What is BNS 318?', or 'Is a non-compete clause valid?'</em>"
            ),
            "category": "fallback",
            "confidence": 0.0,
            "method": "default",
            "found": False
        }

    def get_response(self, query: str) -> dict:
        return self.find_best_match(query)


# Global instance
nlp_chatbot = NPLChatbot()

def get_response(user_question: str) -> str:
    """Returns HTML formatted chatbot response."""
    result = nlp_chatbot.get_response(user_question)
    return result["response"]

def get_response_with_details(user_question: str) -> dict:
    """Returns response with confidence and category metadata."""
    return nlp_chatbot.get_response(user_question)
