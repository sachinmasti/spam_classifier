# Spam Classifier

> AI-powered SMS spam detection with real-time web UI and REST API.

## Overview

This project classifies SMS/email messages as **spam** or **ham** (legitimate) using a machine learning pipeline. Trained on a combined dataset, it extracts linguistic and structural features to detect unwanted messages with high accuracy.

## How It Works

```
User Message
    |
    v
[Feature Extraction] --> text, char_count, digit_count, uppercase_words,
                         has_currency, has_url, has_phone_num
    |
    v
[ML Pipeline]
    |-- CountVectorizer (word frequency)
    |-- TF-IDF Vectorizer (term importance, max 5000 features)
    |-- StandardScaler (normalize numeric features)
    |-- SMOTE (handle class imbalance)
    |-- LightGBM Classifier (gradient boosting)
    |
    v
[Prediction] --> Spam / Ham + Confidence %
```

### Feature Extraction (app/schemas.py)

| Feature | Description |
|---------|-------------|
| `char_count` | Total character length |
| `digit_count` | Number of digits in message |
| `uppercase_words` | Words in ALL CAPS (length > 1) |
| `has_currency` | Contains $, £, €, ¥, or ₹ |
| `has_url` | Contains http, www, .com, etc. |
| `has_phone_num` | Contains 5-12 digit number |

### Web Interface

- Dark theme glassmorphism UI with animated background
- Real-time classification with confidence meter
- Feature breakdown showing which indicators were found
- History log of recent checks
- Mobile responsive

### API

**POST /predict**
```json
// Request
{ "text": "Congratulations! You won a 1000 prize! Call now." }

// Response
{
  "prediction": 1,
  "label": "spam",
  "confidence": 0.98,
  "features": {
    "has_currency": false,
    "has_url": false,
    "has_phone_num": true,
    "uppercase_words": 1,
    "digit_count": 15,
    "char_count": 56
  }
}
```

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.11 | Core programming language |
| **Web Server** | FastAPI + Uvicorn | Async Python web framework + ASGI server |
| **ML Model** | LightGBM | Gradient boosting classifier |
| **Text Vectorization** | TF-IDF + CountVectorizer | Convert text to numerical features |
| **Data Processing** | scikit-learn (Pipeline, ColumnTransformer, StandardScaler) | Feature engineering pipeline |
| **Imbalance Handling** | SMOTE (imbalanced-learn) | Synthetic oversampling for balanced training |
| **Data Handling** | pandas, joblib | Data manipulation, model serialization |
| **Validation** | Pydantic v2 | Request/response schema validation |
| **Frontend** | HTML + CSS + JavaScript (vanilla) | Web UI with glassmorphism design |
| **Containerization** | Docker | Portable deployment |
| **Deployment** | Render (free tier) | Cloud hosting |

## Project Structure

```
spam_classifier/
├── app/
│   ├── app.py              # FastAPI server (routes: /, /predict, /welcome)
│   ├── schemas.py          # Pydantic models + automated feature extraction
│   └── templates/
│       └── index.html      # Web UI with animations
├── model/
│   └── massage_clf.joblib  # Pre-trained LightGBM pipeline
├── training/
│   └── main.py             # Full training pipeline (load, preprocess, train, save)
├── data/
│   └── new_massage_dataset.csv  # Combined spam/ham dataset
├── Dockerfile              # Multi-stage Docker build
├── requirements.txt        # Python dependencies
└── render.yaml             # Render Blueprint config
```

## Quick Start

```bash
# Local
pip install -r requirements.txt
uvicorn app.app:app --host 127.0.0.1 --port 8000

# Docker
docker build -t spam-classifier .
docker run -p 8000:8000 spam-classifier
```

Open http://127.0.0.1:8000

