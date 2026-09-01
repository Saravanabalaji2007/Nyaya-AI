"""
NYAYA AI 3.0 – AI Legal Guardian Chatbot
Uses Google Gemini 2.0 Flash (FREE) as primary engine,
falls back to NLP pattern-matching if Gemini is unavailable.

API: Google Gemini (FREE tier) – https://aistudio.google.com/app/apikey
"""

import re
import os

# ── Primary: Gemini AI chatbot ────────────────────────────────────────────────
GEMINI_CHATBOT_AVAILABLE = False
try:
    from chatbot_gemini import get_response as gemini_get_response, is_gemini_active
    GEMINI_CHATBOT_AVAILABLE = True
    print("✅ Gemini AI chatbot module loaded")
except ImportError as e:
    print(f"⚠️ Gemini chatbot not available ({e}).")

# ── Secondary: NLP-enhanced chatbot ──────────────────────────────────────────
NLP_CHATBOT_AVAILABLE = False
try:
    from chatbot_nlp import get_response_with_details, get_response as nlp_get_response
    NLP_CHATBOT_AVAILABLE = True
    print("✅ NLP-enhanced chatbot loaded successfully")
except ImportError as e:
    print(f"⚠️ NLP chatbot not available ({e}). Using legacy pattern matching.")





# ---------------------------------------------------------------------------
# Knowledge base – keyword patterns → responses
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    # ---- Signed Risky Document – Step-by-step Procedure ----
    {
        "patterns": ["signed risky document", "signed a risky", "risky document signed",
                      "signed bad contract", "signed unfair", "what to do if i signed",
                      "signed risk", "i signed a risky", "signed dangerous document",
                      "signed wrong document", "cheated document", "fraud document"],
        "response": (
            "🚨 <strong>What To Do If You Signed a Risky Legal Document</strong><br><br>"
            "Don't panic! Follow these steps carefully:<br><br>"
            "<strong>📋 Step 1: File an FIR Complaint</strong><br>"
            "• Go to your <strong>nearest police station</strong> immediately.<br>"
            "• File a <strong>First Information Report (FIR)</strong> under the relevant sections.<br>"
            "• Keep a <strong>copy of the FIR</strong> for your records.<br>"
            '<button type="button" class="btn btn-sm btn-outline-primary mt-2 mb-3 px-3 py-1" style="border-radius:20px; font-weight:600;" onclick="askQuestion(\'FIR complaint template\')"><i class="bi bi-file-text me-1"></i> Get FIR Template</button><br>'
            "<strong>📄 Step 2: Gather Evidence</strong><br>"
            "• Collect the <strong>original document</strong> you signed.<br>"
            "• Save all <strong>communications</strong> (emails, messages, call records).<br>"
            "• Note down <strong>witnesses</strong> if any were present during signing.<br><br>"
            "<strong>⚖️ Step 3: Identify the Right Court</strong><br>"
            "What type of document is it?<br>"
            "<div class='d-flex gap-2 mt-2 mb-3 flex-wrap'>"
            "<button type='button' class='btn btn-sm btn-outline-success px-3 py-1' style='border-radius:20px; font-weight:600;' onclick='askQuestion(\"My document involves GST or defective products\")'>GST / Consumer Issue</button>"
            "<button type='button' class='btn btn-sm btn-outline-warning px-3 py-1' style='border-radius:20px; font-weight:600;' onclick='askQuestion(\"My document is a normal agreement or property contract\")'>Normal Contract / Property</button>"
            "</div>"
            "<strong>🏛️ Step 4: Consult a Lawyer</strong><br>"
            "• Based on your document type, consult a specialist advocate.<br>"
            '• Visit our <a href="/lawfirms" style="color:#818cf8;font-weight:600;">Law Firms page</a> to find the right lawyer for your case.<br><br>'
            "💡 <em>Tip: Act quickly! Many legal remedies have time limits (limitation periods). The sooner you act, the better your chances.</em>"
        ),
    },
    # ---- Interactive Court Selection (GST vs Normal) ----
    {
        "patterns": ["gst or defective products", "gst", "defective product", "consumer issue", "consumer document"],
        "response": (
            "🏛️ <strong>Consumer Court</strong><br><br>"
            "Since your document involves GST, defective products, or unfair trade, you should approach the <strong>Consumer Court</strong>.<br><br>"
            "• It is generally faster and cheaper.<br>"
            "• You can file online at <strong>edaakhil.nic.in</strong>.<br><br>"
            '<button type="button" class="btn btn-sm btn-outline-primary mt-2" style="border-radius:20px;" onclick="askQuestion(\'Find lawyer\')"><i class="bi bi-person-badge me-1"></i>Find a Consumer Protection Lawyer</button>'
        ),
    },
    {
        "patterns": ["normal agreement or property", "normal contract", "property contract", "rental agreement", "loan agreement", "service agreement"],
        "response": (
            "🏛️ <strong>Normal Civil Court</strong><br><br>"
            "Since your document is a normal agreement, property contract, or loan, you should file your case in the <strong>Normal Civil Court</strong>.<br><br>"
            "• The jurisdiction depends on the property location or where the agreement was made.<br><br>"
            '<button type="button" class="btn btn-sm btn-outline-primary mt-2" style="border-radius:20px;" onclick="askQuestion(\'Find lawyer\')"><i class="bi bi-person-badge me-1"></i>Find a Civil Legal Lawyer</button>'
        ),
    },
    # ---- FIR Complaint Template ----
    {
        "patterns": ["fir complaint", "fir template", "fir letter", "fir format",
                      "file fir", "first information report", "police complaint",
                      "fir complaint template", "how to file fir"],
        "response": (
            "📝 <strong>FIR Complaint Letter Template</strong><br><br>"
            "<div style='background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.3);border-radius:12px;padding:1.2rem;margin:0.5rem 0;font-family:monospace;font-size:0.88rem;line-height:1.8;'>"
            "<strong>To,</strong><br>"
            "The Station House Officer (SHO),<br>"
            "[Police Station Name],<br>"
            "[City/District, State]<br><br>"
            "<strong>Date:</strong> [DD/MM/YYYY]<br><br>"
            "<strong>Subject:</strong> FIR Complaint Regarding Fraudulent/Risky Legal Document<br><br>"
            "<strong>Respected Sir/Madam,</strong><br><br>"
            "I, <strong>[Your Full Name]</strong>, S/o or D/o <strong>[Father's/Husband's Name]</strong>, "
            "residing at <strong>[Your Complete Address]</strong>, hereby wish to lodge a complaint "
            "regarding the following matter:<br><br>"
            "<strong>Facts of the Case:</strong><br>"
            "1. On <strong>[Date of Signing]</strong>, I entered into an agreement/document with "
            "<strong>[Name of Other Party]</strong>, located at <strong>[Their Address]</strong>.<br>"
            "2. The document titled <strong>[Document Title/Type]</strong> was presented to me under "
            "<strong>[misleading representations / coercion / fraud]</strong>.<br>"
            "3. The document contains the following risky/unfair clauses: <strong>[Describe the risky clauses "
            "– e.g., hidden charges, excessive penalties, waiver of rights, etc.]</strong><br>"
            "4. I have suffered/may suffer financial loss of approximately <strong>₹[Amount]</strong> due to these clauses.<br><br>"
            "<strong>Prayer:</strong><br>"
            "I kindly request you to register an FIR under the appropriate sections of the Indian Penal Code "
            "(IPC Sections 420, 406, 468 as applicable) and take necessary legal action against the accused party.<br><br>"
            "<strong>Yours faithfully,</strong><br>"
            "[Your Full Name]<br>"
            "[Phone Number]<br>"
            "[Email Address]<br>"
            "[Signature]"
            "</div><br>"
            "💡 <em>Tip: Replace all [bracketed] fields with your actual details. Keep 2 photocopies of this complaint. "
            "If the police refuse to file an FIR, you can send it to the Superintendent of Police (SP) by registered post.</em>"
        ),
    },
    # ---- Court Type Guidance ----
    {
        "patterns": ["which court", "consumer court", "civil court", "where to file case",
                      "gst court", "court for gst", "normal court", "district court",
                      "state consumer", "national consumer", "court guidance",
                      "type of court"],
        "response": (
            "🏛️ <strong>Which Court Should You Approach?</strong><br><br>"
            "<strong>📌 Consumer Court (Consumer Dispute Redressal Forum)</strong><br>"
            "File here if your case involves:<br>"
            "• <strong>GST-related fraud</strong> or overcharging<br>"
            "• <strong>Defective products</strong> or deficient services<br>"
            "• <strong>Unfair trade practices</strong> or misleading advertisements<br>"
            "• <strong>E-commerce disputes</strong> (online purchases)<br><br>"
            "<em>Consumer Court Hierarchy:</em><br>"
            "• <strong>District Commission</strong> – Claims up to ₹1 Crore<br>"
            "• <strong>State Commission</strong> – Claims ₹1 Crore to ₹10 Crore<br>"
            "• <strong>National Commission</strong> – Claims above ₹10 Crore<br><br>"
            "<strong>📌 Civil Court (Normal Court)</strong><br>"
            "File here for other matters:<br>"
            "• <strong>Property disputes</strong> and real estate agreements<br>"
            "• <strong>Contract breaches</strong> and loan agreements<br>"
            "• <strong>Employment contract</strong> disputes<br>"
            "• <strong>Partnership deed</strong> disputes<br>"
            "• <strong>Rental/Lease agreement</strong> disputes<br><br>"
            "<em>Civil Court Hierarchy:</em><br>"
            "• <strong>Munsiff Court / Small Causes Court</strong> – Minor civil disputes<br>"
            "• <strong>District Court</strong> – Major civil disputes<br>"
            "• <strong>High Court</strong> – Appeals and writ petitions<br>"
            "• <strong>Supreme Court</strong> – Final appeals<br><br>"
            "💡 <em>Tip: Consumer Court cases are generally faster and cheaper. You can even file a complaint online at "
            "<strong>edaakhil.nic.in</strong>. Consult our "
            '<a href="/lawfirms" style="color:#818cf8;font-weight:600;">recommended lawyers</a> for guidance.</em>'
        ),
    },
    # ---- Case Study Analysis ----
    {
        "patterns": ["case study", "case analysis", "legal case study", "real case",
                      "example case", "case example", "previous case", "court case example",
                      "similar case"],
        "response": (
            "📚 <strong>Legal Case Studies – Document Disputes</strong><br><br>"
            "<strong>📖 Case 1: Hidden Charges in Rental Agreement</strong><br>"
            "<em>Ram vs. Property Dealer (2023)</em><br>"
            "• Ram signed a rental agreement containing hidden maintenance charges of ₹5,000/month.<br>"
            "• He filed a complaint in <strong>Civil Court</strong>.<br>"
            "• <strong>Verdict:</strong> Court ruled the hidden charges were unfair and ordered a refund of ₹60,000 + ₹25,000 compensation.<br>"
            "• <strong>Lesson:</strong> Always read every clause, especially about additional/hidden charges.<br><br>"
            "<strong>📖 Case 2: GST Overcharge on Electronics</strong><br>"
            "<em>Sunita vs. Electronics Store (2024)</em><br>"
            "• Sunita was charged 28% GST on an item that should have been 18%.<br>"
            "• She filed in <strong>District Consumer Court</strong>.<br>"
            "• <strong>Verdict:</strong> Store was ordered to refund the excess GST + ₹10,000 compensation for unfair trade practice.<br>"
            "• <strong>Lesson:</strong> Always verify the GST rate for your product category.<br><br>"
            "<strong>📖 Case 3: Employment Non-Compete Clause</strong><br>"
            "<em>Amit vs. Tech Corp (2023)</em><br>"
            "• Amit signed an employment contract with an overly broad non-compete clause (5 years, all industries).<br>"
            "• He challenged it in <strong>High Court</strong>.<br>"
            "• <strong>Verdict:</strong> Court struck down the clause as unreasonable and against public policy (Section 27 of Indian Contract Act).<br>"
            "• <strong>Lesson:</strong> Excessively broad non-compete clauses are often unenforceable in India.<br><br>"
            "<strong>📖 Case 4: Loan Agreement with Hidden Penalty</strong><br>"
            "<em>Priya vs. Finance Company (2024)</em><br>"
            "• Priya's personal loan agreement had a hidden 5% prepayment penalty clause in fine print.<br>"
            "• She filed in <strong>Banking Ombudsman</strong> and then <strong>Consumer Court</strong>.<br>"
            "• <strong>Verdict:</strong> The hidden penalty was ruled as unfair. Finance company refunded the penalty amount + ₹15,000 compensation.<br>"
            "• <strong>Lesson:</strong> Always scan loan documents for prepayment/penalty clauses.<br><br>"
            "💡 <em>Need help with a similar case? Visit our "
            '<a href="/lawfirms" style="color:#818cf8;font-weight:600;">Law Firms page</a> to find the right advocate.</em>'
        ),
    },
    # ---- Find a Lawyer / Law Firms ----
    {
        "patterns": ["find lawyer", "find advocate", "recommend lawyer", "suggest lawyer",
                      "law firm", "lawfirm", "lawyer near", "advocate near",
                      "lawyer for my case", "best lawyer", "affordable lawyer",
                      "lawyer recommendation"],
        "response": (
            "🏛️ <strong>Find the Right Lawyer</strong><br><br>"
            "We have partnered with <strong>6 specialist advocates</strong> covering all document-related legal fields:<br><br>"
            "• <strong>Contract & Agreement Law</strong> – For rental, service, and partnership agreements<br>"
            "• <strong>Property & Real Estate Law</strong> – For property disputes and land agreements<br>"
            "• <strong>Consumer & GST Protection</strong> – For consumer fraud and GST overcharges<br>"
            "• <strong>Corporate & Commercial Law</strong> – For business contracts and company disputes<br>"
            "• <strong>Employment & Labour Law</strong> – For job contract disputes and wrongful termination<br>"
            "• <strong>Banking & Financial Law</strong> – For loan agreements and banking fraud<br><br>"
            "Our lawyers are classified by affordability:<br>"
            "• 💰 <strong>Middle Class</strong> – ₹12,000 to ₹25,000<br>"
            "• 💰💰 <strong>Upper Middle Class</strong> – ₹18,000 to ₹35,000<br>"
            "• 💰💰💰 <strong>High Class</strong> – ₹40,000 to ₹1,00,000<br><br>"
            '👉 <a href="/lawfirms" style="color:#818cf8;font-weight:600;">Visit our Law Firms page</a> to view all advocates, their fees, specializations, and AI analysis of their expertise.'
        ),
    },
    # ---- Tenant Rights ----
    {
        "patterns": ["tenant right", "tenant rights", "renter right", "rights as a tenant",
                      "tenant protection", "rental rights"],
        "response": (
            "🏠 <strong>Tenant Rights Overview</strong><br><br>"
            "As a tenant, you generally have the following rights:<br>"
            "• <strong>Right to a habitable dwelling</strong> – The landlord must keep the property safe and livable.<br>"
            "• <strong>Right to privacy</strong> – The landlord must provide reasonable notice before entering your home (usually 24-48 hours).<br>"
            "• <strong>Security deposit protection</strong> – Your deposit must be returned within the legally specified period after move-out, minus legitimate deductions.<br>"
            "• <strong>Protection from illegal eviction</strong> – A landlord cannot lock you out or shut off utilities to force you to leave.<br>"
            "• <strong>Right to repairs</strong> – You can request essential repairs and, in many places, withhold rent if critical issues are not fixed.<br><br>"
            "💡 <em>Tip: Always read your rental agreement carefully before signing and keep a copy for your records.</em>"
        ),
    },
    # ---- Rent Increase ----
    {
        "patterns": ["rent increase", "raise rent", "increase my rent", "rent hike"],
        "response": (
            "📈 <strong>Rent Increase Rules</strong><br><br>"
            "• A landlord usually must give <strong>written notice</strong> (30-90 days) before increasing rent.<br>"
            "• Rent cannot be increased during a <strong>fixed-term lease</strong> unless the lease allows it.<br>"
            "• In rent-controlled areas, increases are limited by law.<br>"
            "• An <strong>automatic rent increase</strong> clause means rent goes up without negotiation – watch out for this!<br><br>"
            "💡 <em>Tip: If the increase seems unreasonable, check local rent control laws or consult a legal aid organization.</em>"
        ),
    },
    # ---- Employment Contract ----
    {
        "patterns": ["employment contract", "job contract", "employment agreement",
                      "work contract", "employment risk", "job agreement"],
        "response": (
            "💼 <strong>Employment Contract Risks</strong><br><br>"
            "Watch out for these clauses in your employment contract:<br>"
            "• <strong>Non-compete clause</strong> – May prevent you from working in the same industry after leaving.<br>"
            "• <strong>Termination clause</strong> – Check the notice period required from both sides.<br>"
            "• <strong>Intellectual property assignment</strong> – Your employer may own anything you create, even outside work hours.<br>"
            "• <strong>Liability clause</strong> – You could be held responsible for damages during your employment.<br>"
            "• <strong>Probation period</strong> – You may have fewer rights during this time.<br><br>"
            "💡 <em>Tip: Never sign an employment contract without reading it fully. Ask for clarification on vague terms.</em>"
        ),
    },
    # ---- Loan Agreement ----
    {
        "patterns": ["loan agreement", "loan contract", "loan risk", "borrowing agreement",
                      "loan rules", "personal loan", "home loan", "loan terms"],
        "response": (
            "🏦 <strong>Loan Agreement Rules</strong><br><br>"
            "Key things to check in any loan agreement:<br>"
            "• <strong>Interest rate</strong> – Is it fixed or variable? Variable rates can increase significantly.<br>"
            "• <strong>Prepayment penalty</strong> – Some loans charge a fee if you pay off early.<br>"
            "• <strong>Late payment fees</strong> – Know the penalty for missed payments.<br>"
            "• <strong>Collateral requirements</strong> – Understand what assets are at risk if you default.<br>"
            "• <strong>Hidden charges</strong> – Processing fees, insurance requirements, and administrative costs can add up.<br><br>"
            "💡 <em>Tip: Compare offers from multiple lenders and always calculate the total cost of the loan, not just monthly payments.</em>"
        ),
    },
    # ---- Consumer Protection ----
    {
        "patterns": ["consumer protection", "consumer rights", "buyer rights",
                      "consumer law", "product return", "refund policy", "warranty"],
        "response": (
            "🛡️ <strong>Consumer Protection Laws</strong><br><br>"
            "As a consumer, you are protected by law in most countries:<br>"
            "• <strong>Right to information</strong> – Sellers must disclose product details, price, and terms clearly.<br>"
            "• <strong>Right to safety</strong> – Products must meet safety standards.<br>"
            "• <strong>Right to refund/replacement</strong> – Defective products can be returned or replaced.<br>"
            "• <strong>Protection against unfair trade practices</strong> – Misleading ads, hidden charges, and false claims are illegal.<br>"
            "• <strong>Right to be heard</strong> – You can file complaints with consumer courts or forums.<br><br>"
            "💡 <em>Tip: Always keep receipts, warranty cards, and screenshots of online orders as proof of purchase.</em>"
        ),
    },
    # ---- Penalty Clause ----
    {
        "patterns": ["penalty clause", "penalty in contract", "what is penalty",
                      "penalty meaning"],
        "response": (
            "⚠️ <strong>Penalty Clauses Explained</strong><br><br>"
            "A penalty clause requires you to pay extra money if you break a term in the agreement.<br><br>"
            "• <strong>Common examples</strong>: Early termination fees, late payment charges, breach fines.<br>"
            "• <strong>Legality</strong>: In many jurisdictions, excessive penalties are unenforceable.<br>"
            "• <strong>What to do</strong>: Negotiate penalty amounts before signing. Ask for a cap on penalties.<br><br>"
            "💡 <em>Tip: If a penalty seems unreasonably high, consult a lawyer before signing.</em>"
        ),
    },
    # ---- Eviction ----
    {
        "patterns": ["eviction", "evict me", "eviction notice", "illegal eviction",
                      "can landlord evict"],
        "response": (
            "🚪 <strong>Eviction Rights</strong><br><br>"
            "• A landlord <strong>cannot</strong> evict you without proper legal notice and due process.<br>"
            "• <strong>Illegal eviction</strong> includes changing locks, shutting off utilities, or removing belongings without a court order.<br>"
            "• You have the right to <strong>contest an eviction</strong> in court.<br>"
            "• During the notice period, you can often <strong>cure the violation</strong> (e.g., pay overdue rent) to stop the eviction.<br><br>"
            "💡 <em>Tip: Document everything and seek legal aid if you receive an eviction notice.</em>"
        ),
    },
    # ---- Security Deposit ----
    {
        "patterns": ["security deposit", "deposit refund", "get deposit back",
                      "non-refundable deposit", "deposit return"],
        "response": (
            "💰 <strong>Security Deposit Guidelines</strong><br><br>"
            "• Landlords must return your deposit within the <strong>legally required time</strong> (usually 15-60 days after move-out).<br>"
            "• Deductions are only allowed for <strong>unpaid rent, damages beyond normal wear and tear</strong>, or agreed-upon fees.<br>"
            "• A <strong>non-refundable deposit</strong> means you will NOT get this money back – read carefully before agreeing.<br>"
            "• You can <strong>dispute unfair deductions</strong> through small claims court.<br><br>"
            "💡 <em>Tip: Take photos/videos of the property at move-in and move-out for evidence.</em>"
        ),
    },
    # ---- General Legal Help ----
    {
        "patterns": ["legal help", "need a lawyer", "legal advice",
                      "legal assistance", "legal aid"],
        "response": (
            "⚖️ <strong>Getting Legal Help</strong><br><br>"
            "• <strong>Legal Aid Organizations</strong> offer free help for people who cannot afford a lawyer.<br>"
            "• <strong>Bar Association</strong> referral services can connect you with qualified attorneys.<br>"
            "• <strong>Online legal platforms</strong> provide affordable consultations.<br>"
            "• Many lawyers offer a <strong>free initial consultation</strong>.<br><br>"
        ),
    },
    # ---- Fundamental Rights: Articles 14, 19, 21 ----
    {
        "patterns": ["indian constitution", "basic rights", "fundamental rights", "article 14", "article 19", "article 21", "constitution rights", "constitution"],
        "response": (
            "🏛️ <strong>Indian Constitution - Fundamental Rights</strong><br><br>"
            "<strong>1. Simple Explanation:</strong><br>"
            "The Constitution of India guarantees essential human rights to all citizens, protecting you against unfair state action, discrimination, and unlawful detention.<br><br>"
            "<strong>2. Relevant Indian Laws:</strong><br>"
            "• <strong>Article 14:</strong> Equality before the law (no discrimination based on religion, race, caste, sex).<br>"
            "• <strong>Article 19:</strong> Protection of certain rights regarding freedom of speech, expression, and profession.<br>"
            "• <strong>Article 21:</strong> Right to Protection of Life and Personal Liberty. <em>(Loophole: This article limits arbitrary police arrests).</em><br><br>"
            "<strong>3. Legal Actions:</strong><br>"
            "• You can file a Writ Petition in the High Court under Article 226, or the Supreme Court under Article 32, if these Fundamental Rights are violated.<br>"
            "• Consult a constitutional lawyer via our portal."
        ),
    },
    # ---- IPC Sections: Fraud & Cheating ----
    {
        "patterns": ["ipc", "indian penal code", "cheating", "fraud", "420", "section 420", "section 406", "breach of trust"],
        "response": (
            "⚖️ <strong>IPC Frauds - Cheating & Trust Breach</strong><br><br>"
            "<strong>1. Simple Explanation:</strong><br>"
            "If someone intentionally deceives you to take your money or property, or if someone misuses property you entrusted to them, it is a criminal offense in India.<br><br>"
            "<strong>2. Relevant Indian Laws:</strong><br>"
            "• <strong>IPC Section 420:</strong> Cheating and dishonestly inducing delivery of property (punishable by up to 7 years in prison & fine).<br>"
            "• <strong>IPC Section 406:</strong> Criminal breach of trust (punishable by up to 3 years in prison & fine).<br>"
            "• <em>Loophole: Civil contracts are often disguised as criminal cheating to force police action. Ensure the intent to deceive existed at the very beginning of the contract.</em><br><br>"
            "<strong>3. Legal Actions:</strong><br>"
            "• Lodge an FIR immediately at your local police station.<br>"
            "• If the police refuse to register the FIR, file a private complaint before a Magistrate under CrPC Section 156(3) / BNSS equivalent."
        ),
    },
    # ---- Legal Loopholes: Rental Agreements ----
    {
        "patterns": ["rental loophole", "loophole of rent", "rent loophole", "lease loophole", "11 month agreement", "rent agreement trap"],
        "response": (
            "🏠 <strong>Legal Loopholes - Rental Agreements</strong><br><br>"
            "<strong>1. Simple Explanation:</strong><br>"
            "Landlords often use specific tricks in rental agreements to avoid legal obligations and easily evict tenants or retain security deposits unfairly.<br><br>"
            "<strong>2. Relevant Indian Laws:</strong><br>"
            "• <strong>Registration Act, 1908:</strong> Requires registration of rental agreements longer than 1 year.<br>"
            "• <strong>The Loophole (11-Month Contract):</strong> Landlords almost always create \"Leave and License\" agreements exactly for 11 months. This legally bypasses mandatory registration and heavy stamp duty, effectively stripping you of the protections offered by Rent Control Acts.<br>"
            "• <strong>The Lock-in Clause Trap:</strong> Landlords may hide a lock-in period forcing you to pay rent even if you vacate early for valid reasons.<br><br>"
            "<strong>3. Legal Actions:</strong><br>"
            "• Never verbally agree to term extensions. Insist on a written renewal.<br>"
            "• If evicted illegally by force, file an injunction suit in the Civil Court to restore possession."
        ),
    },
    # ---- Legal Loopholes: Employment Contracts ----
    {
        "patterns": ["employment loophole", "job loophole", "non compete", "notice period trap", "bond loophole", "job contract trick", "employee bond"],
        "response": (
            "💼 <strong>Legal Loopholes - Employment Contracts</strong><br><br>"
            "<strong>1. Simple Explanation:</strong><br>"
            "Companies often insert scary clauses into employment letters that are designed to intimidate employees from leaving, holding back their salaries or preventing them from joining competitors. Most of these are actually illegal.<br><br>"
            "<strong>2. Relevant Indian Laws:</strong><br>"
            "• <strong>Indian Contract Act, 1872 (Section 27):</strong> Agreements in restraint of trade (Non-Compete clauses) are void in India.<br>"
            "• <strong>The Loophole (Non-Compete):</strong> HR departments put \"2-year non-compete clauses\" in contracts knowing they have zero legal validity in an Indian court after an employee resigns. It is purely a psychological trap.<br>"
            "• <strong>The Loophole (Employment Bonds):</strong> Companies cannot force you to pay massive \"training bonds\" or withhold your relieving letter if you quit, unless they can absolutely prove they spent that exact amount on specialized external training for you.<br><br>"
            "<strong>3. Legal Actions:</strong><br>"
            "• Send a formal legal notice for the release of your final settlement and experience letter.<br>"
            "• File a complaint with the Labour Commissioner or approach the Civil Court for damages."
        ),
    },
    # ---- Legal Loopholes: Loan & Finance Contracts ----
    {
        "patterns": ["loan loophole", "finance loophole", "bank loophole", "arbitration trap", "blank cheque trap", "hidden penalty", "loan traps", "loan trap"],
        "response": (
            "🏦 <strong>Legal Loopholes - Loan Agreements</strong><br><br>"
            "<strong>1. Simple Explanation:</strong><br>"
            "Private lenders and finance companies bury hidden traps in the fine print to guarantee they win legal disputes and extract maximum penalties from borrowers.<br><br>"
            "<strong>2. Relevant Indian Laws:</strong><br>"
            "• <strong>Arbitration and Conciliation Act, 1996.</strong><br>"
            "• <strong>Negotiable Instruments Act, 1881 (Section 138 - Cheque Bounce).</strong><br>"
            "• <strong>The Loophole (Unilateral Arbitration):</strong> The loan agreement will state that any dispute must be solved by an Arbitrator chosen solely by the bank. This ensures a heavily biased judgment against the borrower without ever going to a real court.<br>"
            "• <strong>The Loophole (Security Cheques):</strong> Lenders demand undated, blank signed cheques as \"security\". If you miss a payment, they fill in a massive inflated default amount and bounce it, weaponizing Section 138 (a criminal offense) against you to extort money.<br><br>"
            "<strong>3. Legal Actions:</strong><br>"
            "• Approach the Banking Ombudsman (RBI) to complain about unfair practices.<br>"
            "• File a Consumer Court complaint against unfair trade contracts and contest the unilateral appointment of an arbitrator."
        ),
    },
    # ---- General Legal Help / Fallback (Replacing standard help with precise formatted one) ----
    {
        "patterns": ["help", "what can you do", "features", "options", "general rights", "fir complaint", "consumer court"],
        "response": (
            "⚖️ <strong>NYAYA AI - Indian Legal Expert</strong><br><br>"
            "<strong>1. Simple Explanation:</strong><br>"
            "I am programmed with highly specific expertise on fundamental Indian laws and common drafting loopholes used to trap citizens.<br><br>"
            "<strong>2. Supported Indian Laws & Topics:</strong><br>"
            "• <strong>Indian Constitution:</strong> (Ask about \"Article 14, 19, 21\")<br>"
            "• <strong>Indian Penal Code:</strong> (Ask about \"IPC Fraud, section 420, breach of trust\")<br>"
            "• <strong>Common Loopholes:</strong> (Ask about \"rental loophole\", \"employment bond trick\", or \"loan arbitration trap\")<br>"
            "• <strong>Basic Rights:</strong> (Ask about \"tenant rights\" or \"consumer protection\")<br><br>"
            "<strong>3. Legal Actions:</strong><br>"
            "• Type one of the topics above to see the precise loopholes and remedies!"
        ),
    }
]

# Greeting patterns
GREETING_PATTERNS = ["hello", "hi", "hey", "good morning", "good afternoon",
                     "good evening", "namaste", "greetings"]


def get_response(user_message: str, history: list = None) -> str:
    """
    Get response using:
    1. Google Gemini 2.0 Flash AI (FREE tier) – primary
    2. NLP-enhanced chatbot – secondary fallback
    3. Legacy pattern matching – tertiary fallback
    """
    # ── 1. Gemini AI (primary) ────────────────────────────────────────────────
    if GEMINI_CHATBOT_AVAILABLE:
        try:
            return gemini_get_response(user_message, history=history)
        except Exception as e:
            print(f"Gemini chatbot error: {e}. Trying NLP...")

    # ── 2. NLP chatbot (secondary) ────────────────────────────────────────────
    if NLP_CHATBOT_AVAILABLE:
        try:
            return nlp_get_response(user_message)
        except Exception as e:
            print(f"NLP chatbot error: {e}. Falling back to legacy mode.")

    # ── 3. Legacy pattern matching (tertiary) ─────────────────────────────────
    msg = user_message.lower().strip()

    # Check greetings first
    for greet in GREETING_PATTERNS:
        if greet in msg:
            return (
                "👋 <strong>Hello!</strong> I'm <strong>NYAYA AI Legal Guardian</strong>.<br><br>"
                "I can help you with:<br>"
                "• 🚨 What to do if you signed a risky document<br>"
                "• 📝 FIR complaint template & filing guidance<br>"
                "• ⚖️ Which court to approach (Consumer vs Civil)<br>"
                "• 📚 Legal case studies & examples<br>"
                "• 🏠 Tenant rights & rent rules<br>"
                "• 💼 Employment contract risks<br>"
                "• 🏦 Loan agreement terms<br>"
                "• 🛡️ Consumer protection laws<br>"
                "• 🏛️ Find a specialist lawyer<br><br>"
                "Just type your question and I'll do my best to help! 😊"
            )

    # Search knowledge base
    for entry in KNOWLEDGE_BASE:
        for pattern in entry["patterns"]:
            if pattern in msg:
                return entry["response"]

    # Fallback response
    return (
        "🤔 I'm not sure I understand that question yet.<br><br>"
        "I am specifically trained as an elite expert on Indian Constitutional Rights, the IPC, and Legal Loopholes.<br><br>"
        "💡 <strong>Try asking about my specialties:</strong><br>"
        "• <strong>Constitution:</strong> <em>\"What are my fundamental rights under Article 21?\"</em><br>"
        "• <strong>IPC Fraud:</strong> <em>\"What is IPC section 420 cheating?\"</em><br>"
        "• <strong>Job Loopholes:</strong> <em>\"Is an employment bond loophole legal?\"</em><br>"
        "• <strong>Rent Loopholes:</strong> <em>\"What is the 11-month rent agreement trap?\"</em><br>"
        "• <strong>Loan Traps:</strong> <em>\"What is the arbitration loan loophole?\"</em><br>"
        "• <strong>General rights:</strong> <em>\"FIR complaint\"</em> or <em>\"Consumer Court\"</em>"
    )

