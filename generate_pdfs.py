import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Normal / Safe Document (Fair Standard Agreement)
# ---------------------------------------------------------------------------
safe_text = [
    "THIS RESIDENTIAL LEASE & SERVICE AGREEMENT is executed on this 15th day of August, 2026 at Chennai, Tamil Nadu.",
    "BETWEEN: Mr. Ramesh Kumar (hereinafter referred to as the 'Lessor/Owner') AND Mr. Suresh Sharma (hereinafter referred to as the 'Lessee/Tenant').",
    "",
    "WHEREAS the Lessor is the absolute lawful owner of the premises located at Flat 4B, Green Avenue, Anna Nagar, Chennai - 600040.",
    "AND WHEREAS the Lessee desires to rent the aforesaid residential premises for residential occupation.",
    "",
    "NOW THIS AGREEMENT WITNESSETH AS FOLLOWS:",
    "",
    "## 1. Lease Term & Rent Payment",
    "- The tenure of this lease agreement shall be for a period of 11 months, commencing from September 1, 2026.",
    "- The monthly rent agreed upon is Rs. 18,000 (Eighteen Thousand Rupees only), payable on or before the 5th of every calendar month.",
    "- Rent increases shall be capped at 5% per annum upon mutual agreement at the time of lease renewal.",
    "",
    "## 2. Refundable Security Deposit",
    "- The Lessee has deposited a refundable security sum of Rs. 50,000 (Fifty Thousand Rupees only) with the Lessor.",
    "- The entire deposit sum of Rs. 50,000 shall be refunded to the Lessee within 15 days of peaceful handover of premises.",
    "- Deductions shall be made strictly for actual unpaid utility bills or structural damage beyond fair wear and tear.",
    "",
    "## 3. Maintenance, Repairs & Utilities",
    "- The Lessor shall be solely responsible for all major structural repairs, plumbing leaks, and building structural upkeep.",
    "- The Lessee shall pay routine electricity and water consumption charges as billed directly by government utilities.",
    "- Routine minor maintenance under Rs. 500 shall be attended to by the Lessee in a timely manner.",
    "",
    "## 4. Termination & Notice Period",
    "- Either party may terminate this agreement by providing a clear 30-day prior written notice.",
    "- No arbitrary termination penalties or forfeiture of deposits shall apply during or after the notice period.",
    "",
    "## 5. Dispute Resolution & Jurisdiction",
    "- Any dispute arising under this agreement shall be settled amicably between the parties within 30 days.",
    "- If unresolved, disputes shall be referred to the competent Civil Court of jurisdiction in Chennai, Tamil Nadu."
]

# ---------------------------------------------------------------------------
# 2. Risk Document (High Risk / Fraudulent / Abusive Agreement)
# ---------------------------------------------------------------------------
risky_text = [
    "THIS BINDING COMMERCIAL & RESIDENTIAL SERVICE CONTRACT is entered into on August 15, 2026.",
    "BY AND BETWEEN: Apex Realty & Capital Services Pvt. Ltd. (the 'Company') AND the undersigned Client/Tenant ('Signatory').",
    "",
    "WARNING: READ CAREFULLY. THIS CONTRACT CONTAINS BINDING PENALTIES, WAIVERS OF RIGHTS, AND UNILATERAL CHARGES.",
    "",
    "## 1. Fees, Hidden Charges & Compounding Penalties",
    "- The base monthly rate is set at Rs. 25,000. However, the Company reserves the absolute right to impose unlimited hidden charges, administrative fees, and technology levies without prior notice.",
    "- Any payment delayed beyond 24 hours will automatically trigger an immediate compounding penalty of 10% per day on the total outstanding balance.",
    "- The Signatory agrees that all fee structures are non-negotiable and non-refundable under any circumstances.",
    "",
    "## 2. Automatic Rent Increase & Deposit Forfeiture",
    "- The contract features an automatic rent increase of 20% every 3 months, executed unilaterally by the Company.",
    "- The Signatory must pay a non-refundable deposit of Rs. 1,00,000. Upon contract expiration or early exit, the Signatory shall automatically forfeit the full security deposit.",
    "",
    "## 3. Excessive Cancellation & Termination Fees",
    "- In the event the Signatory attempts to terminate or cancel this agreement early, an exorbitant termination fee of Rs. 75,000 shall be levied immediately.",
    "- The Company reserves the right to lock the premises and impound all personal possessions until all fees and penalties are cleared in full.",
    "",
    "## 4. Total Liability Waiver & Indemnity",
    "- The Signatory accepts total liability for all accidents, fire, theft, or property loss, even if caused directly by the Company's gross negligence.",
    "- The Signatory completely waives all rights to sue the Company, its directors, or affiliates in any court of law.",
    "",
    "## 5. Restraint of Business & Non-Compete Trap",
    "- The Signatory is strictly prohibited from working, leasing, or operating any business in the same district for a period of 3 years following termination.",
    "",
    "## 6. Unilateral Arbitration Clause",
    "- Any dispute shall be decided exclusively by a sole arbitrator appointed by the Company. All legal and tribunal costs shall be borne 100% by the Signatory."
]

def make_pdf(filename, title, lines):
    filepath = os.path.join(OUT, filename)
    doc = SimpleDocTemplate(
        filepath, 
        pagesize=A4,
        topMargin=0.8*inch, 
        bottomMargin=0.8*inch,
        leftMargin=0.8*inch, 
        rightMargin=0.8*inch
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Title'], fontSize=16, leading=20, spaceAfter=18
    )
    body_style = ParagraphStyle(
        'CustomBody', parent=styles['Normal'], fontSize=10.5, leading=15, spaceAfter=8
    )
    
    story = [Paragraph(title, title_style), Spacer(1, 10)]
    
    for line in lines:
        if line.startswith("##"):
            story.append(Spacer(1, 8))
            story.append(Paragraph(f"<b>{line.replace('## ', '')}</b>", body_style))
        elif line.startswith("-"):
            story.append(Paragraph(f"&bull; {line[1:].strip()}", body_style))
        elif line == "":
            story.append(Spacer(1, 4))
        else:
            story.append(Paragraph(line, body_style))
            
    doc.build(story)
    print(f"[PDF Created] {filepath}")

def make_txt(filename, title, lines):
    filepath = os.path.join(OUT, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        for line in lines:
            f.write(line + "\n")
    print(f"[TXT Created] {filepath}")

if __name__ == "__main__":
    # Generate main safe agreement (both txt and pdf)
    make_pdf("safe_agreement.pdf", "Standard Fair Legal Agreement (Normal Document)", safe_text)
    make_txt("safe_agreement.txt", "Standard Fair Legal Agreement (Normal Document)", safe_text)
    make_pdf("safe_agreement_v2.pdf", "Standard Fair Legal Agreement (Normal Document)", safe_text)
    make_txt("safe_agreement_v2.txt", "Standard Fair Legal Agreement (Normal Document)", safe_text)
    
    # Generate main risk agreement (both txt and pdf)
    make_pdf("risky_agreement.pdf", "High Risk Legal Agreement (Risk Document)", risky_text)
    make_txt("risky_agreement.txt", "High Risk Legal Agreement (Risk Document)", risky_text)
    make_pdf("risky_agreement_v2.pdf", "High Risk Legal Agreement (Risk Document)", risky_text)
    make_txt("risky_agreement_v2.txt", "High Risk Legal Agreement (Risk Document)", risky_text)
    make_txt("moderate_risk_agreement.txt", "Moderate Risk Legal Agreement", safe_text[:10] + risky_text[6:10])
    
    print("\nAll Normal and Risk sample documents generated successfully in TXT and PDF formats!")

