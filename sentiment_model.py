import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("student_feedback.csv")

X = df["feedback"]
y = df["sentiment"]

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

model_sent = MultinomialNB()
model_sent.fit(X_vec, y)

# Save model & vectorizer
with open("models/sentiment_model.pkl", "wb") as f:
    pickle.dump(model_sent, f)

with open("models/sentiment_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Sentiment model saved!")
