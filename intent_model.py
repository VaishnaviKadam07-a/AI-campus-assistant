import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("sustainability_qa.csv")

X = df["question"]
y = df["intent"]

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

model_intent = MultinomialNB()
model_intent.fit(X_vec, y)

with open("models/intent_model.pkl", "wb") as f:
    pickle.dump(model_intent, f)

with open("models/intent_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Intent model saved!")
