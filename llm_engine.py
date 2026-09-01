"""
NYAYA AI 3.0 – LLM Engine for Document Analysis & Chatbot
Uses Hugging Face Transformers (flan-t5-small for Q&A, facebook/bart-large-cnn for summarization).
Models are loaded lazily on first use. Falls back gracefully if unavailable.
"""

import re
import traceback

# ---------------------------------------------------------------------------
# Lazy-load LLM models (downloaded once, cached locally)
# ---------------------------------------------------------------------------
LLM_AVAILABLE = False
_summarizer = None
_qa_pipeline = None
_qa_tokenizer = None
_qa_model = None

LEGAL_CONTEXT = """You are NYAYA AI, an expert Indian legal assistant specializing in:
- Indian Constitution (Articles 14, 19, 21)
- Indian Penal Code (IPC Sections 420, 406, 468)
- Contract law, tenant rights, employment law, consumer protection
- Legal loopholes in rental, employment, and loan agreements
- Filing FIRs, approaching courts (Consumer Court vs Civil Court)
Answer in a helpful, clear, and structured manner with practical legal advice."""


def _load_summarizer():
    """Load the summarization model (BART) – lazy initialization."""
    global _summarizer, LLM_AVAILABLE
    if _summarizer is not None:
        return _summarizer
    try:
        from transformers import pipeline
        print("🔄 Loading summarization model (facebook/bart-large-cnn)...")
        _summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1,  # CPU
            framework="pt"
        )
        LLM_AVAILABLE = True
        print("✅ Summarization model loaded successfully")
        return _summarizer
    except Exception as e:
        print(f"⚠️ Could not load summarization model: {e}")
        return None


def _load_qa_model():
    """Load the text-generation model (flan-t5-small) – lazy initialization."""
    global _qa_pipeline, _qa_tokenizer, _qa_model, LLM_AVAILABLE
    if _qa_pipeline is not None:
        return _qa_pipeline
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
        print("🔄 Loading Q&A model (google/flan-t5-small)...")
        model_name = "google/flan-t5-small"
        _qa_tokenizer = T5Tokenizer.from_pretrained(model_name)
        _qa_model = T5ForConditionalGeneration.from_pretrained(model_name)
        _qa_pipeline = pipeline(
            "text2text-generation",
            model=_qa_model,
            tokenizer=_qa_tokenizer,
            device=-1,  # CPU
            max_new_tokens=256
        )
        LLM_AVAILABLE = True
        print("✅ Q&A model loaded successfully")
        return _qa_pipeline
    except Exception as e:
        print(f"⚠️ Could not load Q&A model: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Public API Functions
# ---------------------------------------------------------------------------

def summarize_document(text: str, max_length: int = 200, min_length: int = 50) -> str:
    """
    Generate a plain-English summary of a legal document using BART LLM.
    
    Args:
        text: The full document text
        max_length: Maximum summary length in tokens
        min_length: Minimum summary length in tokens
    
    Returns:
        A string summary, or None if LLM is unavailable
    """
    summarizer = _load_summarizer()
    if summarizer is None:
        return _fallback_summarize(text)
    
    try:
        # BART has a max input of ~1024 tokens, truncate long docs
        # Approximate: 1 token ≈ 4 chars
        max_chars = 3500
        input_text = text[:max_chars] if len(text) > max_chars else text
        
        # Ensure minimum input length for summarization
        if len(input_text.split()) < 30:
            return _fallback_summarize(text)
        
        result = summarizer(
            input_text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )
        
        summary = result[0]["summary_text"]
        return summary
    except Exception as e:
        print(f"⚠️ Summarization error: {e}")
        traceback.print_exc()
        return _fallback_summarize(text)


def analyze_clauses_llm(text: str, clauses: list) -> list:
    """
    Use LLM to provide detailed plain-English explanations of risky clauses.
    
    Args:
        text: Full document text
        clauses: List of matched risky clauses from the risk scanner
    
    Returns:
        List of dicts with 'clause', 'llm_explanation' fields
    """
    qa = _load_qa_model()
    if qa is None:
        return []
    
    analyzed = []
    for clause in clauses[:5]:  # Limit to 5 clauses to avoid slow processing
        keyword = clause.get("keyword", "")
        sentence = clause.get("sentence", "")
        
        if not sentence:
            continue
        
        try:
            prompt = (
                f"Explain this legal clause in simple English for a common person. "
                f"What does it mean and what risk does it pose?\n\n"
                f"Clause: \"{sentence}\"\n\n"
                f"Simple explanation:"
            )
            
            result = qa(prompt, max_new_tokens=150)
            explanation = result[0]["generated_text"].strip()
            
            analyzed.append({
                "clause": sentence,
                "keyword": keyword,
                "llm_explanation": explanation,
            })
        except Exception as e:
            print(f"⚠️ Clause analysis error for '{keyword}': {e}")
    
    return analyzed


def chat_with_llm(question: str, context: str = "") -> dict:
    """
    Generate an LLM-powered response for legal questions.
    Used as fallback when NLP pattern matching has low confidence.
    
    Args:
        question: User's legal question
        context: Optional context (e.g., from uploaded documents)
    
    Returns:
        dict with 'response', 'method', 'confidence' fields, or None if unavailable
    """
    qa = _load_qa_model()
    if qa is None:
        return None
    
    try:
        prompt = (
            f"{LEGAL_CONTEXT}\n\n"
            f"Question: {question}\n\n"
        )
        
        if context:
            prompt += f"Context from document: {context[:500]}\n\n"
        
        prompt += "Answer:"
        
        result = qa(prompt, max_new_tokens=256)
        response_text = result[0]["generated_text"].strip()
        
        if not response_text or len(response_text) < 10:
            return None
        
        # Format the response with HTML for the chatbot UI
        formatted_response = _format_llm_response(response_text, question)
        
        return {
            "response": formatted_response,
            "raw_response": response_text,
            "method": "llm",
            "confidence": 0.7,
            "model": "flan-t5-small"
        }
    except Exception as e:
        print(f"⚠️ LLM chat error: {e}")
        traceback.print_exc()
        return None


def get_document_insights(text: str) -> dict:
    """
    Generate comprehensive document insights using all LLM capabilities.
    
    Returns:
        dict with 'summary', 'document_type', 'key_points'
    """
    insights = {
        "summary": None,
        "document_type": _detect_document_type(text),
        "key_points": _extract_key_points(text),
        "llm_powered": False,
    }
    
    # Get LLM summary
    summary = summarize_document(text)
    if summary:
        insights["summary"] = summary
        insights["llm_powered"] = True
    
    return insights


# ---------------------------------------------------------------------------
# Fallback / Helper Functions
# ---------------------------------------------------------------------------

def _fallback_summarize(text: str) -> str:
    """Simple extractive summary when LLM is not available."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= 3:
        return text.strip()
    
    # Take first 3 sentences as summary
    summary = " ".join(sentences[:3])
    if len(summary) > 500:
        summary = summary[:497] + "..."
    return summary


def _detect_document_type(text: str) -> str:
    """Detect the type of legal document based on keywords."""
    text_lower = text.lower()
    
    doc_types = {
        "Rental Agreement": ["rent", "tenant", "landlord", "lease", "premises"],
        "Employment Contract": ["employment", "employee", "employer", "salary", "designation", "probation"],
        "Loan Agreement": ["loan", "borrower", "lender", "interest rate", "repayment", "principal"],
        "Sale Deed": ["sale", "seller", "buyer", "property", "consideration", "conveyance"],
        "Service Agreement": ["service provider", "client", "scope of work", "deliverables"],
        "Partnership Deed": ["partnership", "partner", "profit sharing", "firm"],
        "Insurance Policy": ["insurance", "policy", "premium", "claim", "insured"],
    }
    
    best_match = "General Legal Document"
    best_score = 0
    
    for doc_type, keywords in doc_types.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_match = doc_type
    
    return best_match if best_score >= 2 else "General Legal Document"


def _extract_key_points(text: str) -> list:
    """Extract key points from document text using NLP heuristics."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    important_indicators = [
        "shall", "must", "required", "obligated", "penalty",
        "terminate", "liable", "forfeit", "waive", "indemnify",
        "non-refundable", "binding", "irrevocable", "automatic",
        "consent", "agree", "guarantee", "warranty"
    ]
    
    key_points = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        importance = sum(1 for ind in important_indicators if ind in sentence_lower)
        if importance >= 1 and len(sentence) > 20:
            key_points.append(sentence.strip())
    
    return key_points[:8]  # Return top 8 key points


def _format_llm_response(response: str, question: str) -> str:
    """Format LLM response with HTML for chatbot display."""
    # Clean up the response
    response = response.strip()
    
    # Add structure
    formatted = (
        f"🤖 <strong>AI-Generated Legal Guidance</strong><br><br>"
        f"{response}<br><br>"
        f"<small class='text-muted'><i class='bi bi-robot me-1'></i>"
        f"This response was generated by AI (flan-t5). For critical legal matters, "
        f"please consult a qualified lawyer.</small>"
    )
    
    return formatted


# ---------------------------------------------------------------------------
# Module-level status check
# ---------------------------------------------------------------------------
def check_llm_status() -> dict:
    """Check the status of LLM models without loading them."""
    return {
        "summarizer_loaded": _summarizer is not None,
        "qa_loaded": _qa_pipeline is not None,
        "llm_available": LLM_AVAILABLE,
    }


print("✅ LLM engine module initialized (models will load on first use)")
