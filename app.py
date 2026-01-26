import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
#from rag_chatbot import get_rag_answer


# ============================
# LOAD MODELS & VECTORIZERS
# ============================

# Eco Classification Model
with open("models/eco_model.pkl", "rb") as f:
    eco_model = pickle.load(f)

with open("models/eco_encoders.pkl", "rb") as f:
    eco_encoders = pickle.load(f)

# Sentiment Analysis Model
with open("models/sentiment_model.pkl", "rb") as f:
    sentiment_model = pickle.load(f)

with open("models/sentiment_vectorizer.pkl", "rb") as f:
    sentiment_vectorizer = pickle.load(f)

# Intent Classification Model
with open("models/intent_model.pkl", "rb") as f:
    intent_model = pickle.load(f)

with open("models/intent_vectorizer.pkl", "rb") as f:
    intent_vectorizer = pickle.load(f)

# ============================
# STREAMLIT CONFIG
# ============================

st.set_page_config(
    page_title="AI-Powered Green Campus Assistant",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 AI-Powered Green Campus Assistant")

menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Eco Score", "💬 Chatbot", "📝 Feedback"]
)

# ============================
# HOME
# ============================

if menu == "🏠 Home":
    st.header("Welcome Student 👋")
    st.write("""
    This AI-powered assistant promotes sustainable campus habits by:
    - 🌿 Tracking eco-friendly actions
    - 💬 Answering sustainability questions
    - 🧠 Analyzing student feedback using AI
    """)

# ============================
# ECO SCORE DASHBOARD
# ============================

elif menu == "📊 Eco Score":
    st.header("📊 Eco-Friendly Activity Tracker")

    electricity = st.selectbox("Electricity Usage", ["Low", "Medium", "High"])
    water = st.selectbox("Water Saving", ["Yes", "No"])
    plastic = st.selectbox("Plastic Usage", ["Low", "Medium", "High"])
    transport = st.selectbox("Transport Mode", ["Walk", "Bicycle", "Bus", "Bike"])
    recycling = st.selectbox("Recycling Habit", ["Yes", "No"])

    if st.button("Check Eco Friendliness"):
        input_df = pd.DataFrame([{
            "electricity_usage": electricity,
            "water_saving": water,
            "plastic_usage": plastic,
            "transport_mode": transport,
            "recycling": recycling
        }])

        # Encode categorical features
        for col in input_df.columns:
            input_df[col] = eco_encoders[col].transform(
                input_df[col].values.reshape(-1, 1)
            )

        prediction = eco_model.predict(input_df)[0]

        # Eco score logic
        score_map = {
            "Low": 3,
            "Medium": 2,
            "High": 1,
            "Yes": 3,
            "No": 1,
            "Walk": 3,
            "Bicycle": 3,
            "Bus": 2,
            "Bike": 1
        }

        eco_score = (
            score_map[electricity] +
            score_map[water] +
            score_map[plastic] +
            score_map[transport] +
            score_map[recycling]
        )

        if prediction == 1:
            st.success("🌱 You are Eco-Friendly! Keep it up!")
        else:
            st.error("⚠ You can improve your eco-friendly habits.")

        st.metric("🌿 Eco Score", eco_score)

        # Chart
        st.subheader("📈 Eco Behavior Breakdown")

        chart_df = pd.DataFrame({
            "Category": ["Electricity", "Water", "Plastic", "Transport", "Recycling"],
            "Score": [
                score_map[electricity],
                score_map[water],
                score_map[plastic],
                score_map[transport],
                score_map[recycling]
            ]
        })

        fig, ax = plt.subplots()
        ax.bar(chart_df["Category"], chart_df["Score"])
        ax.set_ylabel("Score")
        ax.set_title("Sustainability Contribution")

        st.pyplot(fig)

# ============================
# CHATBOT (INTENT MODEL)
# ============================

elif menu == "💬 Chatbot":
    st.header("💬 RAG-Based Sustainability Chatbot")

    user_question = st.text_input("Ask about campus sustainability:")

    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Thinking... 🌱"):
                answer = get_rag_answer(user_question)
            st.success(answer)
        else:
            st.warning("Please enter a question.")

# ============================
# FEEDBACK (SENTIMENT ANALYSIS)

# ============================

elif menu == "📝 Feedback":
    st.header("📝 Student Feedback")

    feedback = st.text_area("Write your feedback:")

    if st.button("Submit Feedback"):
        if feedback:
            vec = sentiment_vectorizer.transform([feedback])
            sentiment = sentiment_model.predict(vec)[0]

            st.write(f"📊 Sentiment Detected: **{sentiment}**")

            if sentiment == "Positive":
                st.success("Thank you! We're glad you like the initiative 🌱")
            elif sentiment == "Neutral":
                st.info("Thanks! We'll work on improving the experience.")
            else:
                st.error("Thanks for the feedback. We'll try to improve.")
        else:
            st.warning("Please write feedback before submitting.")
