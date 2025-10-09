import streamlit as st
import pandas as pd
import pickle
from transformers import pipeline
from utils import clean_text, generate_reply, format_summary  # ✅ Import helper functions

# ----------------------------
# ⚙️ Load Model & Vectorizer
# ----------------------------
@st.cache_resource
def load_model():
    """Load the trained TF-IDF vectorizer and classification model."""
    with open("model.pkl", "rb") as f:
        vectorizer, model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_model()

# ----------------------------
# 🧠 Load Summarization Model
# ----------------------------
@st.cache_resource
def load_summarizer():
    """Load the Hugging Face BART summarization pipeline."""
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# ----------------------------
# 🌟 Streamlit UI Configuration
# ----------------------------
st.set_page_config(page_title="AI Mail Assistant", page_icon="🤖", layout="centered")

st.title("🤖 AI Mail Insight Assistant")
st.caption("Built with Machine Learning + Generative AI (SymphonyAI-Style Project)")

# ----------------------------
# 🧠 Initialize Session State
# ----------------------------
if "session_results" not in st.session_state:
    st.session_state.session_results = pd.DataFrame(columns=["Email", "Category", "Summary", "AI_Reply"])

# ----------------------------
# ✉️ User Input
# ----------------------------
user_input = st.text_area(
    "✉️ Paste your email content here:",
    height=150,
    placeholder="Example: Hi IT team, my laptop is not connecting to Wi-Fi..."
)

# ----------------------------
# 🚀 Email Analysis Logic
# ----------------------------
if st.button("Analyze Email"):
    if user_input.strip() == "":
        st.warning("Please enter an email to analyze.")
    else:
        # ✅ Step 1: Clean the email text
        cleaned_text = clean_text(user_input)

        # ✅ Step 2: Predict category using ML model
        category = model.predict(vectorizer.transform([cleaned_text]))[0]

        # ✅ Step 3: Summarize the email using GenAI
        summary_result = summarizer(user_input, max_length=40, min_length=10, do_sample=False)
        summary = format_summary(summary_result[0]['summary_text'])

        # ✅ Step 4: Generate polite AI reply based on category
        ai_reply = generate_reply(category)

        # ✅ Step 5: Display results in UI
        st.success("✅ Email Analyzed Successfully!")
        st.markdown(f"**📂 Predicted Category:** `{category}`")
        st.markdown(f"**🧠 Summary:** {summary}")
        st.markdown(f"**💌 Suggested AI Reply:** {ai_reply}")

        # ✅ Step 6: Save result to session log
        new_entry = pd.DataFrame([[user_input, category, summary, ai_reply]],
                                 columns=["Email", "Category", "Summary", "AI_Reply"])
        st.session_state.session_results = pd.concat(
            [st.session_state.session_results, new_entry],
            ignore_index=True
        )

        # ✅ Step 7: Save the latest entry to a separate CSV
        new_entry.to_csv("latest_result.csv", index=False)

# ----------------------------
# 📋 Show Session Log
# ----------------------------
if not st.session_state.session_results.empty:
    st.subheader("📊 Session Log (All Analyzed Emails)")
    st.dataframe(st.session_state.session_results, use_container_width=True)

    # ✅ Allow full session export
    csv_all = st.session_state.session_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download All Session Results as CSV",
        data=csv_all,
        file_name="all_email_analyses.csv",
        mime="text/csv",
    )

# ----------------------------
# 🏁 Footer Section
# ----------------------------
st.divider()
st.caption("Created by Harshitha • AI Engineer Candidate • SymphonyAI Internship Project")
