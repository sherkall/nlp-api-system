from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session
import nltk
import string
import pickle

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from database import Base, engine, get_db, TextRecord
from auth import (
    User, hash_password, verify_password,
    create_access_token, get_current_user
)

app = FastAPI(title="Text Processing Service")

Base.metadata.create_all(bind=engine)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

with open("model/sentiment_model.pkl", "rb") as f:
    sentiment_model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


class TextInput(BaseModel):
    text: str


class RegisterInput(BaseModel):
    username: str
    password: str


@app.get("/")
def home():
    return {"message": "Text Processing Service is running"}


@app.post("/register")
def register(data: RegisterInput, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == data.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")
    user = User(username=data.username, hashed_password=hash_password(data.password))
    db.add(user)
    db.commit()
    return {"message": f"User '{data.username}' registered successfully"}


@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/clean")
def clean_text(data: TextInput, current_user: User = Depends(get_current_user)):
    text = data.text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return {"original": data.text, "cleaned": text}


@app.post("/tokenize")
def tokenize_text(data: TextInput, current_user: User = Depends(get_current_user)):
    tokens = word_tokenize(data.text.lower())
    return {"original": data.text, "tokens": tokens}


@app.post("/preprocess")
def preprocess_text(
    data: TextInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    text = data.text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    final = " ".join(tokens)

    record = TextRecord(original_text=data.text, processed_text=final)
    db.add(record)
    db.commit()

    return {
        "original": data.text,
        "cleaned": text,
        "tokens_after_stopword_removal": tokens,
        "final_processed": final
    }


@app.post("/predict")
def predict_sentiment(
    data: TextInput,
    current_user: User = Depends(get_current_user)
):
    text = data.text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    processed = " ".join(tokens)

    features = vectorizer.transform([processed])
    prediction = sentiment_model.predict(features)[0]
    confidence = sentiment_model.predict_proba(features)[0]

    label = "positive" if prediction == 1 else "negative"
    score = round(float(max(confidence)) * 100, 1)

    return {
        "original": data.text,
        "processed": processed,
        "sentiment": label,
        "confidence": f"{score}%"
    }


@app.get("/history")
def get_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    records = db.query(TextRecord).order_by(TextRecord.timestamp.desc()).all()
    return [
        {
            "id": r.id,
            "original": r.original_text,
            "processed": r.processed_text,
            "timestamp": r.timestamp
        }
        for r in records
    ]