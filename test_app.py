"""
NYAYA AI 3.0 – Flask Test Suite
Tests all routes, authentication, document upload, ML land prediction,
TN land verification, chatbot, and translation.
"""

import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import io
import json
from app import app, init_db

print("=" * 65)
print("  NYAYA AI 3.0 – Flask Application Test Suite")
print("=" * 65)

# Initialize database
init_db()
client = app.test_client()

# 1. Landing Page
print("\n[1] Testing Landing Page...")
res = client.get('/')
assert res.status_code == 200, f"Expected 200, got {res.status_code}"
print("  ✓ Landing Page OK (200)")

# 2. Signup & Login
print("\n[2] Testing Authentication (Signup & Login)...")
test_email = "nyayatest@example.com"
client.post('/signup', data={"name": "Nyaya Tester", "email": test_email, "password": "securepass123"})
login_res = client.post('/login', data={"email": test_email, "password": "securepass123"}, follow_redirects=True)
assert login_res.status_code == 200, f"Expected 200, got {login_res.status_code}"
print("  ✓ User Authentication OK")

# 3. ML Land Price Prediction
print("\n[3] Testing ML Land Price Prediction API (/predict_land)...")
pred_res = client.post('/predict_land', json={
    "district": "Chennai",
    "property_type": "Residential",
    "area": 1,
    "unit": "ground",
    "road_width_ft": 40,
    "guideline_val_sqft": 6500,
    "prev_sale_sqft": 8800
})
assert pred_res.status_code == 200, f"Expected 200, got {pred_res.status_code}"
pred_data = json.loads(pred_res.data)
print(f"  ✓ Predicted Value: {pred_data['prediction']['estimated_market_value_formatted']}")
print(f"  ✓ Guideline Value: {pred_data['prediction']['government_guideline_value_formatted']}")
print(f"  ✓ Confidence Score: {pred_data['prediction']['confidence_score']}%")

# 4. TN Land Ownership Verification
print("\n[4] Testing TN Land Verification API (/verify_land)...")
verify_res = client.post('/verify_land', json={
    "district": "Chennai",
    "patta": "1001",
    "owner": "Muthu Ramanathan",
    "survey": "12/1"
})
assert verify_res.status_code == 200, f"Expected 200, got {verify_res.status_code}"
verify_data = json.loads(verify_res.data)
print(f"  ✓ Verification Status: {verify_data['status']}")
print(f"  ✓ Risk Level: {verify_data.get('risk_level')}")

# 5. Document Upload & Structured Scan
print("\n[5] Testing Document Upload & Structured NYAYA AI 3.0 Scan (/upload)...")
sample_text = (
    "RENTAL AGREEMENT: Tenant shall pay a non-refundable deposit of ₹80,000. "
    "A mandatory lock-in period of 11 months applies. "
    "Late payment attracts a 10% daily penalty. "
    "Maintenance charges of ₹4,000 per month will be added."
)
data = {
    'document': (io.BytesIO(sample_text.encode('utf-8')), 'test_agreement.txt')
}
upload_res = client.post('/upload', data=data, content_type='multipart/form-data')
assert upload_res.status_code == 200, f"Expected 200, got {upload_res.status_code}"
assert b"NYAYA AI 3.0" in upload_res.data or b"Document Summary" in upload_res.data, "Structured report not found in response HTML"
print("  ✓ Document Upload & Structured Report OK (200)")

# 6. Chatbot API
print("\n[6] Testing Chatbot API (/chat_api)...")
chat_res = client.post('/chat_api', json={"question": "What is BNS Section 318 for cheating?"})
assert chat_res.status_code == 200, f"Expected 200, got {chat_res.status_code}"
chat_data = json.loads(chat_res.data)
assert "response" in chat_data and len(chat_data["response"]) > 10, "Empty chat response"
print("  ✓ Chatbot API OK (200)")

# 7. Translation API
print("\n[7] Testing Translation API (/translate)...")
trans_res = client.post('/translate', json={"text": "Hello, this is a legal test", "target": "ta", "source": "en"})
assert trans_res.status_code == 200, f"Expected 200, got {trans_res.status_code}"
trans_data = json.loads(trans_res.data)
print(f"  ✓ Translated Result: {trans_data.get('translated')}")

print("\n" + "=" * 65)
print("  🎉 All 7 Flask Application Tests Passed Successfully!")
print("=" * 65)
