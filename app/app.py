from app.schemas import Message
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import joblib
from pathlib import Path

import pandas as pd

app = FastAPI(title="Spam Classifier")
MODEL_PATH = Path(__file__).parent.parent / "model" / "massage_clf.joblib"
INDEX_HTML = Path(__file__).parent / "templates" / "index.html"

model = joblib.load(MODEL_PATH)


@app.get('/', response_class=HTMLResponse)
def home():
    return HTMLResponse(INDEX_HTML.read_text(encoding='utf-8'))


@app.get('/welcome')
def welcome() -> dict:
    return {'message': 'Welcome to our website'}


@app.post('/predict')
def predict(message: Message):
    data = message.model_dump()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    proba = model.predict_proba(df)[0]

    features = {
        'has_currency': bool(data.get('has_currency', 0)),
        'has_url': bool(data.get('has_url', 0)),
        'has_phone_num': bool(data.get('has_phone_num', 0)),
        'uppercase_words': data.get('uppercase_words', 0),
        'digit_count': data.get('digit_count', 0),
        'char_count': data.get('char_count', 0),
    }

    return {
        'prediction': int(prediction[0]),
        'label': 'spam' if int(prediction[0]) == 1 else 'ham',
        'confidence': round(float(proba[int(prediction[0])]), 4),
        'features': features,
    }