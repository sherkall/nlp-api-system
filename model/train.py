import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

data = {
    "text": [
        # Positive
        "I love this product, it is amazing",
        "This is the best thing I have ever used",
        "Absolutely fantastic experience",
        "I am so happy with this purchase",
        "Great quality and fast delivery",
        "Highly recommend this to everyone",
        "This exceeded all my expectations",
        "Wonderful service and great value",
        "I am very satisfied with the results",
        "This works perfectly and I love it",
        "Outstanding performance and quality",
        "Best purchase I have made this year",
        "Really impressed with the quality",
        "Superb experience from start to finish",
        "Excellent product highly recommended",
        # Negative
        "This is terrible and a waste of money",
        "I hate this product it is awful",
        "Worst experience I have ever had",
        "Very disappointed with the quality",
        "This broke after one day of use",
        "Do not buy this complete rubbish",
        "Absolutely horrible will not recommend",
        "Poor quality and terrible service",
        "I am very unhappy with this purchase",
        "This product is a complete disaster",
        "Dreadful experience avoid at all costs",
        "Terrible waste of time and money",
        "Very poor quality not worth it",
        "Awful product broke immediately",
        "Disgusting quality never buying again",
    ],
    "label": [1]*15 + [0]*15
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.1f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

os.makedirs("model", exist_ok=True)
with open("model/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model saved to model/sentiment_model.pkl")
print("Vectorizer saved to model/vectorizer.pkl")