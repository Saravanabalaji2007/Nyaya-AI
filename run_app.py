#!/usr/bin/env python
"""Simple script to run the Flask app"""
import sys
import os

# Ensure UTF-8 output encoding for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Load .env file (GEMINI_API_KEY and other env vars)
def _load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip()
                    if key and value and not value.startswith('replace-with'):
                        os.environ.setdefault(key, value)
        print("✅ Environment variables loaded from .env")

_load_dotenv()

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Import and run the app
from app import app, init_db

if __name__ == "__main__":
    init_db()
    print("\n" + "=" * 55)
    print("  NYAYA AI 3.0 - AI Legal Guardian is running!")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 55 + "\n")
    app.run(debug=True, host="127.0.0.1", port=5000)
