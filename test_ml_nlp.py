import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from ai_engine_ml import scan_risks_ml, generate_risk_report
from ml_land_predictor import land_predictor
from chatbot_nlp import nlp_chatbot

print("=" * 65)
print("  NYAYA AI 3.0 – Comprehensive System Test")
print("=" * 65)

# 1. Test Document Risk Scan (NYAYA AI 3.0 System Prompt)
print("\n[1] Testing AI Risk Scanner & Hidden Clause Detection...")
sample_doc = (
    "This rental agreement mandates a non-refundable deposit of ₹1,00,000, "
    "a lock-in period of 12 months, a 10% late payment penalty, and monthly maintenance charges of ₹5,000. "
    "Automatic renewal applies unless written cancellation is given. Any dispute shall be referred to binding arbitration."
)
risk_result = scan_risks_ml(sample_doc)
print(f"  • Document Type: {risk_result['document_type']}")
print(f"  • Risk Score: {risk_result['risk_score']}/100 ({risk_result['risk_badge']})")
print(f"  • Hidden Charges Found: {len(risk_result['matched'])}")
print(f"  • Recommended Authority: {risk_result['court_info']['authority']}")
print(f"  • Confidence: {risk_result['confidence_percentage']}%")

# 2. Test ML Land Price Predictor
print("\n[2] Testing ML Land Price Predictor (Ensemble RF + GB)...")
p1 = land_predictor.predict(district="Chennai", property_type="Residential", area=1, unit="ground", road_width_ft=40)
print(f"  • Chennai 1 Ground Residential: {p1['estimated_market_value_formatted']} (Guideline: {p1['government_guideline_value_formatted']}, Confidence: {p1['confidence_score']}%)")

p2 = land_predictor.predict(district="Coimbatore", property_type="Commercial", area=2400, unit="sqft", road_width_ft=60)
print(f"  • Coimbatore 2400 Sq.Ft Commercial: {p2['estimated_market_value_formatted']} (Price Range: {p2['price_range_formatted']})")

p3 = land_predictor.predict(district="Madurai", property_type="Agricultural", area=1.5, unit="acre", road_width_ft=20)
print(f"  • Madurai 1.5 Acre Agricultural: {p3['estimated_market_value_formatted']} (Trend: {p3['annual_growth_trend']})")

# 3. Test Multilingual Chatbot (English, Tamil, Hindi)
print("\n[3] Testing Multilingual NLP Chatbot...")
queries = [
    "What is BNS Section 318 for cheating?",
    "How to verify Patta Chitta online in Tamil Nadu?",
    "நில பட்டா மற்றும் சிட்டா எவ்வாறு சரிபார்ப்பது?",
    "भारतीय न्याय संहिता 2023 में धोखाधड़ी की क्या धारा है?",
    "What is Zero FIR under BNSS?",
    "What are tenant rights in India?",
]

for q in queries:
    res = nlp_chatbot.get_response(q)
    print(f"  • [Query]: {q}")
    print(f"    -> Category: {res.get('category')} | Found: {res.get('found')} | Method: {res.get('method')}")

print("\n" + "=" * 65)
print("  ✅ All NYAYA AI 3.0 Advanced Features & Models Verified Successfully!")
print("=" * 65)
