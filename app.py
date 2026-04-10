import streamlit as st
import pickle
import re

# Page config
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

# Load model
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text

# 🔥 DARK THEME CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}

.block-container {
    background-color: rgba(0, 0, 0, 0.7);
    padding: 2rem;
    border-radius: 15px;
}

h1 {
    text-align: center;
    color: #00c6ff;
}

p {
    text-align: center;
    color: #dcdcdc;
}

/* Text Area */
.stTextArea textarea {
    background-color: #1e1e1e;
    color: white;
    border-radius: 10px;
}

/* Button */
.stButton>button {
    width: 100%;
    border-radius: 10px;
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white;
    font-size: 16px;
    padding: 10px;
    border: none;
}

/* Success & Error Box */
.stSuccess {
    background-color: #1b5e20 !important;
    color: white !important;
}

.stError {
    background-color: #b71c1c !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# UI
st.title("📰 Fake News Detection")
st.write("Check whether a news article is REAL or FAKE using NLP")

news = st.text_area("Enter News Article", height=200)

if st.button("🔍 Analyze News"):

    if news.strip() == "":
        st.warning("Please enter some text!")
    else:
        cleaned = clean_text(news)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.success("✅ This is REAL NEWS")
        else:
            st.error("❌ This is FAKE NEWS")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;'>Developed using NLP & Machine Learning</p>", unsafe_allow_html=True)
