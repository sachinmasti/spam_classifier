<div align="center">

# 🛡️ Spam Classifier

### AI-powered SMS/Email spam detection with a real-time web UI and REST API

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-success)](https://lightgbm.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render&logoColor=white)](https://spam-classifier-ffg1.onrender.com)

**[🚀 Try the Live Demo](https://spam-classifier-ffg1.onrender.com)**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [How It Works](#-how-it-works)
- [Features](#-features)
- [API Reference](#-api-reference)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Author](#-author)

---

## 🔍 Overview

This project classifies SMS and email messages as **spam** or **ham** (legitimate) using a complete machine learning pipeline. It extracts linguistic and structural features from raw text, runs them through a tuned LightGBM classifier, and serves predictions through both a FastAPI REST endpoint and a polished web interface — all containerized and deployed live.

## 🌐 Live Demo

The app is live and ready to test — no setup required:

**👉 [https://spam-classifier-ffg1.onrender.com](https://spam-classifier-ffg1.onrender.com)**

> ⚠️ Hosted on Render's free tier — the server may take a few seconds to spin up if it's been idle.

## ⚙️ How It Works

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

## ✨ Features

### Feature Extraction (`app/schemas.py`)

| Feature | Description |
|---|---|
| `char_count` | Total character length |
| `digit_count` | Number of digits in message |
| `uppercase_words` | Words in ALL CAPS (length > 1) |
| `has_currency` | Contains $, £, €, ¥, or ₹ |
| `has_url` | Contains http, www, .com, etc. |
| `has_phone_num` | Contains 5–12 digit number |

### Web Interface

- 🎨 Dark theme glassmorphism UI with animated background
- ⚡ Real-time classification with a confidence meter
- 🔬 Feature breakdown showing which indicators were found
- 🕓 History log of recent checks
- 📱 Mobile responsive

## 📡 API Reference

**`POST /predict`**

```json
// Request
{ "text": "Congratulations! You won a 1000 prize! Call now." }
```

```json
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

**Quick test with curl:**

```bash
curl -X POST https://spam-classifier-ffg1.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You won a 1000 prize! Call now."}'
```

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
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
| **Deployment** | Render | Cloud hosting |

## 📁 Project Structure

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

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- pip
- Docker (optional, for containerized run)

### Local Setup

```bash
git clone https://github.com/sachinmasti/spam_classifier.git
cd spam_classifier
pip install -r requirements.txt
uvicorn app.app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**

### Run with Docker

```bash
docker build -t spam-classifier .
docker run -p 8000:8000 spam-classifier
```


## 👤 Author

**Sachin Masti**

- GitHub: [@sachinmasti](https://github.com/sachinmasti)
- Blog: [Sachinmasti on Medium](https://medium.com/@Sachinmasti)

---

<div align="center">

If you found this project useful, consider giving it a ⭐!

</div>
