# Nyaya AI  — AI Legal Guardian

Nyaya AI is a local web application that helps users review legal documents, identify potentially risky clauses, ask legal questions through a chatbot, and explore land-price predictions.

> **Important:** Nyaya AI is an informational tool. It does not replace advice from a qualified legal professional.

## Features

- **Document risk analysis** — upload PDF, DOCX, or TXT legal documents and identify clauses that may need attention.
- **Clause explanations** — simplify selected legal language into clearer wording.
- **Legal chatbot** — ask general legal questions using the built-in NLP chatbot, with optional Gemini-powered responses.
- **Land-price prediction** — estimate land prices using the included trained machine-learning model.
- **Local data storage** — uses SQLite, so no database server is required.

## Requirements

- Python 3.10 or later
- Internet access only if you choose to use Gemini-powered chatbot responses

## Installation

1. Clone or download this repository.

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Optional: copy `.env.example` to `.env` and replace the placeholder with your Gemini API key.

4. Start the application:

   ```bash
   python run_app.py
   ```

5. Open `http://127.0.0.1:5000` in your browser.

On Windows, you can instead run `run_nyaya_ai.bat`.

## Project structure

```text
Nyaya-AI/
├── app.py                    # Flask application and routes
├── run_app.py                # Application launcher
├── ai_engine.py              # Document risk-analysis logic
├── chatbot*.py               # Chatbot and NLP components
├── ml_land_predictor.py      # Land-price prediction logic
├── models/                   # Pre-trained ML models
├── datasets/                 # Legal Q&A dataset
├── templates/                # HTML pages
├── static/                   # Stylesheets and images
├── sample_docs/              # Example legal documents
├── uploads/                  # Documents uploaded at runtime
├── nyaya_ai.db               # Local SQLite database
├── requirements.txt          # Python dependencies
└── .env.example              # Safe environment-variable template
```

