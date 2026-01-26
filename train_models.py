import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("student_eco_actions.csv")

# Separate encoders for each column
encoders = {}
categorical_cols = [
    "electricity_usage",
    "water_saving",
    "plastic_usage",
    "transport_mode",
    "recycling"
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Create label
df["eco_label"] = df["eco_score"].apply(lambda x: 1 if x >= 10 else 0)

X = df[categorical_cols]
y = df["eco_label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
eco_model = RandomForestClassifier(n_estimators=100, random_state=42)
eco_model.fit(X_train, y_train)

# Evaluate
y_pred = eco_model.predict(X_test)
print("Eco Model Accuracy:", accuracy_score(y_test, y_pred))

# Save model & encoders
with open("models/eco_model.pkl", "wb") as f:
    pickle.dump(eco_model, f)

with open("models/eco_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("Eco model & encoders saved successfully!")
