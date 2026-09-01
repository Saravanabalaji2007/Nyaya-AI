"""
NYAYA AI 3.0 – NLP Chatbot Trainer
Trains the chatbot using the Indian Legal Q&A dataset with:
- TF-IDF Vectorization for semantic understanding with Multilingual Support (English, Tamil, Hindi)
- Cosine Similarity for question matching
- Naive Bayes classifier for intent/category classification
- Pickle-based model persistence for ultra-fast loading

Dataset: datasets/indian_legal_qa.csv (Kaggle Indian Legal Texts + LawSum + ILDC + TN Acts)
"""

import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import csv
import pickle
import re
import numpy as np
from pathlib import Path

# ML/NLP Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data
for resource in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = BASE_DIR / "datasets" / "indian_legal_qa.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

TFIDF_MODEL_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
QA_VECTORS_PATH = MODEL_DIR / "qa_vectors.pkl"
QA_DATA_PATH = MODEL_DIR / "qa_data.pkl"
CATEGORY_MODEL_PATH = MODEL_DIR / "category_classifier.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"


# ---------------------------------------------------------------------------
# Multilingual Text Preprocessing
# ---------------------------------------------------------------------------
stemmer = PorterStemmer()

def preprocess_text(text: str) -> str:
    """Clean and preprocess text for NLP with Multilingual (EN, TA, HI) support."""
    text = text.lower().strip()
    # Remove punctuation while preserving Unicode alphanumeric characters (Tamil, Hindi, English)
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    
    try:
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        # Retain key interrogatives and legal keywords
        legal_keep = {'what', 'how', 'when', 'where', 'which', 'who', 'can', 'is', 'are', 'not', 'no', 'why'}
        processed = []
        for t in tokens:
            if (t not in stop_words or t in legal_keep) and len(t) > 1:
                # Only stem English words
                if t.isascii() and t.isalpha():
                    processed.append(stemmer.stem(t))
                else:
                    processed.append(t)
        return ' '.join(processed) if processed else text
    except Exception:
        return text


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------
def load_dataset(path: str = None) -> list:
    """Load the Indian Legal Q&A dataset from CSV."""
    if path is None:
        path = DATASET_PATH
    
    dataset = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('question') and row.get('answer'):
                    dataset.append({
                        'question': row['question'].strip(),
                        'answer': row['answer'].strip(),
                        'category': row.get('category', 'general').strip(),
                        'source': row.get('source', 'Unknown').strip(),
                    })
        print(f"[Dataset] Loaded {len(dataset)} Q&A pairs from dataset")
    except FileNotFoundError:
        print(f"[Warning] Dataset not found at {path}")
    except Exception as e:
        print(f"[Error] Error loading dataset: {e}")
    
    return dataset


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------
def train_models(dataset: list = None) -> dict:
    """
    Train NLP models on the legal Q&A dataset.
    
    Returns:
        dict with trained models and data
    """
    if dataset is None:
        dataset = load_dataset()
    
    if not dataset:
        print("[Warning] No dataset available for training")
        return None
    
    print(f"\n[Training] Training NLP models on {len(dataset)} Q&A pairs...")
    
    # Prepare data
    questions = [item['question'] for item in dataset]
    answers = [item['answer'] for item in dataset]
    categories = [item['category'] for item in dataset]
    sources = [item['source'] for item in dataset]
    
    # Preprocess questions for TF-IDF
    processed_questions = [preprocess_text(q) for q in questions]
    
    # ============ 1. TF-IDF Vectorizer for Question Matching ============
    print("  [Step 1] Training TF-IDF Vectorizer with Multilingual N-Grams...")
    tfidf = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        token_pattern=r'(?u)\b\w+\b'  # Support all Unicode words
    )
    qa_vectors = tfidf.fit_transform(processed_questions)
    print(f"     Vocabulary size: {len(tfidf.vocabulary_)}")
    print(f"     Feature matrix: {qa_vectors.shape}")
    
    # ============ 2. Category Classifier (Naive Bayes) ============
    print("  [Step 2] Training Category Classifier...")
    label_encoder = LabelEncoder()
    encoded_categories = label_encoder.fit_transform(categories)
    
    category_clf = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=4000, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b')),
        ('clf', MultinomialNB(alpha=0.1))
    ])
    category_clf.fit(questions, encoded_categories)
    print(f"     Categories: {list(label_encoder.classes_)}")
    
    # ============ 3. Save Models ============
    print("  [Step 3] Saving trained models...")
    
    with open(TFIDF_MODEL_PATH, 'wb') as f:
        pickle.dump(tfidf, f)
    
    with open(QA_VECTORS_PATH, 'wb') as f:
        pickle.dump(qa_vectors, f)
    
    qa_data = {
        'questions': questions,
        'answers': answers,
        'categories': categories,
        'sources': sources,
        'processed_questions': processed_questions,
    }
    with open(QA_DATA_PATH, 'wb') as f:
        pickle.dump(qa_data, f)
    
    with open(CATEGORY_MODEL_PATH, 'wb') as f:
        pickle.dump(category_clf, f)
    
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"  [Saved] Models saved to {MODEL_DIR}")
    
    result = {
        'tfidf': tfidf,
        'qa_vectors': qa_vectors,
        'qa_data': qa_data,
        'category_clf': category_clf,
        'label_encoder': label_encoder,
        'num_samples': len(dataset),
        'num_categories': len(label_encoder.classes_),
    }
    
    print(f"\n[Complete] Training complete!")
    print(f"   [Data] {len(dataset)} Q&A pairs trained")
    print(f"   [Categories] {len(label_encoder.classes_)} categories: {list(label_encoder.classes_)}")
    print(f"   [Features] TF-IDF features: {qa_vectors.shape[1]}")
    
    return result


def load_trained_models() -> dict:
    """Load pre-trained models from disk."""
    try:
        with open(TFIDF_MODEL_PATH, 'rb') as f:
            tfidf = pickle.load(f)
        with open(QA_VECTORS_PATH, 'rb') as f:
            qa_vectors = pickle.load(f)
        with open(QA_DATA_PATH, 'rb') as f:
            qa_data = pickle.load(f)
        with open(CATEGORY_MODEL_PATH, 'rb') as f:
            category_clf = pickle.load(f)
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        
        print(f"[Loaded] Loaded trained models ({len(qa_data['questions'])} Q&A pairs)")
        
        return {
            'tfidf': tfidf,
            'qa_vectors': qa_vectors,
            'qa_data': qa_data,
            'category_clf': category_clf,
            'label_encoder': label_encoder,
        }
    except FileNotFoundError:
        print("[Notice] No trained models found. Training new models...")
        return train_models()
    except Exception as e:
        print(f"[Notice] Error loading models: {e}. Retraining...")
        return train_models()


# ---------------------------------------------------------------------------
# Inference / Question Answering
# ---------------------------------------------------------------------------
class TrainedLegalChatbot:
    """NLP chatbot trained on Indian Legal Q&A dataset."""
    
    def __init__(self):
        self.models = None
        self.is_ready = False
        self._initialize()
    
    def _initialize(self):
        """Load or train models."""
        self.models = load_trained_models()
        if self.models:
            self.is_ready = True
    
    def find_answer(self, question: str, threshold: float = 0.12) -> dict:
        """
        Find the best answer for a question using TF-IDF cosine similarity.
        """
        if not self.is_ready:
            return None
        
        tfidf = self.models['tfidf']
        qa_vectors = self.models['qa_vectors']
        qa_data = self.models['qa_data']
        category_clf = self.models['category_clf']
        label_encoder = self.models['label_encoder']
        
        # Preprocess the question
        processed = preprocess_text(question)
        
        # ============ TF-IDF Cosine Similarity ============
        query_vector = tfidf.transform([processed])
        similarities = cosine_similarity(query_vector, qa_vectors)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[-3:][::-1]
        best_idx = top_indices[0]
        best_score = float(similarities[best_idx])
        
        # ============ Category Prediction ============
        try:
            predicted_category_idx = category_clf.predict([question])[0]
            predicted_category = label_encoder.inverse_transform([predicted_category_idx])[0]
        except Exception:
            predicted_category = "general_law"
        
        if best_score >= threshold:
            answer = qa_data['answers'][best_idx]
            matched_question = qa_data['questions'][best_idx]
            category = qa_data['categories'][best_idx]
            source = qa_data['sources'][best_idx]
            
            # Format the answer with HTML
            formatted_answer = self._format_answer(answer, category, source, best_score)
            
            return {
                'answer': formatted_answer,
                'raw_answer': answer,
                'matched_question': matched_question,
                'confidence': best_score,
                'category': category,
                'predicted_category': predicted_category,
                'source': source,
                'method': 'nlp_trained',
                'found': True,
                'top_matches': [
                    {
                        'question': qa_data['questions'][idx],
                        'score': float(similarities[idx]),
                    }
                    for idx in top_indices if similarities[idx] > 0.05
                ]
            }
        
        return {
            'answer': None,
            'confidence': best_score,
            'category': predicted_category,
            'method': 'nlp_trained',
            'found': False,
        }
    
    def _format_answer(self, answer: str, category: str, source: str, confidence: float) -> str:
        """Format the answer with HTML for display."""
        category_icons = {
            'ipc': "<i class='bi bi-scale me-1 text-warning'></i>",
            'constitution': "<i class='bi bi-bank me-1 text-primary'></i>",
            'crpc': "<i class='bi bi-journal-text me-1 text-info'></i>",
            'contract_law': "<i class='bi bi-file-earmark-text me-1 text-success'></i>",
            'consumer_law': "<i class='bi bi-shield-check me-1 text-primary'></i>",
            'tenant_law': "<i class='bi bi-house-door me-1 text-info'></i>",
            'employment_law': "<i class='bi bi-briefcase me-1 text-warning'></i>",
            'family_law': "<i class='bi bi-people me-1 text-secondary'></i>",
            'property_law': "<i class='bi bi-building me-1 text-primary'></i>",
            'cyber_law': "<i class='bi bi-laptop me-1 text-info'></i>",
            'financial_law': "<i class='bi bi-cash-stack me-1 text-success'></i>",
            'criminal_law': "<i class='bi bi-exclamation-triangle me-1 text-danger'></i>",
            'administrative_law': "<i class='bi bi-bank2 me-1 text-primary'></i>",
            'environmental_law': "<i class='bi bi-tree me-1 text-success'></i>",
            'general_law': "<i class='bi bi-book me-1 text-secondary'></i>",
            'case_study': "<i class='bi bi-journal-bookmark me-1 text-warning'></i>",
            'court_guidance': "<i class='bi bi-building-gear me-1 text-info'></i>",
            'general_conversation': "<i class='bi bi-chat-dots me-1 text-success'></i>",
        }
        icon = category_icons.get(category, "<i class='bi bi-book me-1 text-secondary'></i>")
        
        # Split sentences nicely
        paragraphs = re.split(r'(?<=[.!?])\s+', answer)
        formatted_text = '<br>'.join([f"• {p.strip()}" for p in paragraphs if len(p.strip()) > 3])
        if not formatted_text:
            formatted_text = answer
        
        html = (
            f"{icon} <strong>{category.replace('_', ' ').title()}</strong><br><br>"
            f"{formatted_text}<br><br>"
            f"<small style='color: #94a3b8;'>"
            f"<i class='bi bi-book me-1'></i>Source: {source} | Confidence: {int(confidence*100)}%"
            f"</small>"
        )
        
        return html
    
    def get_training_stats(self) -> dict:
        """Return training statistics."""
        if not self.is_ready:
            return {'status': 'not_ready'}
        
        qa_data = self.models['qa_data']
        label_encoder = self.models['label_encoder']
        
        from collections import Counter
        category_counts = Counter(qa_data['categories'])
        
        return {
            'status': 'ready',
            'total_qa_pairs': len(qa_data['questions']),
            'categories': list(label_encoder.classes_),
            'num_categories': len(label_encoder.classes_),
            'category_distribution': dict(category_counts),
            'tfidf_features': self.models['qa_vectors'].shape[1],
        }


# Auto-train if models don't exist
if not TFIDF_MODEL_PATH.exists():
    train_models()

# Create global chatbot instance
trained_chatbot = TrainedLegalChatbot()


if __name__ == "__main__":
    print("=" * 60)
    print("  NYAYA AI 3.0 – NLP Chatbot Training & Verification")
    print("=" * 60)
    
    result = train_models()
    
    if result:
        chatbot = TrainedLegalChatbot()
        test_queries = [
            "What is BNS Section 318 for cheating?",
            "How to verify Patta Chitta online in Tamil Nadu?",
            "What is the landmark Puttaswamy case on Right to Privacy?",
            "What are tenant rights in Tamil Nadu?",
            "நில பட்டா மற்றும் சிட்டா எவ்வாறு சரிபார்ப்பது?",
            "भारतीय न्याय संहिता 2023 में धोखाधड़ी की क्या धारा है?",
            "What is Zero FIR under BNSS?",
            "What are hidden maintenance charges in agreements?",
        ]
        
        print("\n" + "=" * 60)
        print("  Testing Trained Multilingual Chatbot")
        print("=" * 60)
        
        for query in test_queries:
            res = chatbot.find_answer(query)
            if res and res['found']:
                print(f"\n[Q] {query}")
                print(f"   [Confidence]: {res['confidence']:.2f}")
                print(f"   [Category]: {res['category']}")
                print(f"   [Source]: {res['source']}")
                print(f"   [Matched]: {res['matched_question']}")
            else:
                print(f"\n[Q] {query}")
                print(f"   [No Match] (confidence: {res['confidence']:.2f})")
        
        stats = chatbot.get_training_stats()
        print(f"\n[Stats] Total Q&A Pairs: {stats['total_qa_pairs']}, Categories: {stats['num_categories']}, TF-IDF Features: {stats['tfidf_features']}")
