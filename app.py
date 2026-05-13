import streamlit as st
import pandas as pd
import os
import re
import string
import nltk
import numpy as np
import gdown
import joblib

from nltk.corpus import stopwords

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="✨",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: #F0F4F8;
}

.hero {
    background: #00AA5B;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
}

.hero h1 {
    font-family: 'Space Mono', monospace !important;
    color: white !important;
    font-size: 2.2rem;
}

.hero p {
    color: rgba(255,255,255,0.9) !important;
}

.card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

.predict-positive {
    background: #E6F7EE;
    border: 2px solid #00AA5B;
    border-radius: 16px;
    padding: 1rem;
    text-align: center;
}

.predict-negative {
    background: #FFF0ED;
    border: 2px solid #FF6B35;
    border-radius: 16px;
    padding: 1rem;
    text-align: center;
}

.predict-result {
    font-size: 1.2rem;
    font-weight: 800;
}

.stButton > button {
    background: #00AA5B !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

#MainMenu, footer, header {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def setup_nltk():
    nltk.download('stopwords', quiet=True)

setup_nltk()

important_words = {
    'tidak', 'tak', 'bukan', 'belum', 'jangan', 'tanpa',
    'kurang', 'gagal', 'susah', 'sulit', 'males', 'malas',
    'buruk', 'jelek', 'parah', 'lambat', 'lemot', 'bohong'
}

stop_words = set(stopwords.words('indonesian')) - important_words

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 1]
    return " ".join(words)

@st.cache_resource
def load_models():

    model1_id = "1f0OBJGZNhmIUMMqBNOYQOexZrh7XUSkp"
    model1_path = "models_ds1.pkl"

    if not os.path.exists(model1_path):
        url1 = f"https://drive.google.com/uc?id={model1_id}"
        gdown.download(url1, model1_path, quiet=False)

    model2_id = "1fo2n7JHmS8WavNCbOAyUSVlIu4bo3wHF"
    model2_path = "models_ds2.pkl"

    if not os.path.exists(model2_path):
        url2 = f"https://drive.google.com/uc?id={model2_id}"
        gdown.download(url2, model2_path, quiet=False)

    models_ds1 = joblib.load(model1_path)
    models_ds2 = joblib.load(model2_path)

    return {
        "Dataset 1": models_ds1,
        "Dataset 2": models_ds2
    }

models = load_models()

def predict_sentiment(text, model, tfidf):

    cleaned = clean_text(text)
    processed = preprocess_text(cleaned)

    vect = tfidf.transform([processed])

    pred = model.predict(vect)[0]

    return "POSITIVE ✅" if pred == 1 else "NEGATIVE ❌"

st.markdown("""
<div class="hero">
    <h1>Sentiment Analyzer ✨</h1>
    <p>Analyze user reviews to understand public sentiment.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Try Sentiment Analysis")

user_input = st.text_area(
    "Enter a review:",
    placeholder="Example: Aplikasinya bagus dan drivernya ramah",
    height=140
)

dataset_choice = st.selectbox(
    "Choose Dataset Model:",
    ["Dataset 1", "Dataset 2"]
)

predict_btn = st.button(
    "Analyze Sentiment →",
    use_container_width=True
)

if predict_btn:

    if user_input.strip() == "":
        st.warning("Please enter a review first.")

    else:

        selected_models = models[dataset_choice]

        tfidf = selected_models['tfidf']

        nb_model = selected_models['nb']
        lr_model = selected_models['lr']
        svm_model = selected_models['svm']
        rf_model = selected_models['rf']

        nb_pred = predict_sentiment(user_input, nb_model, tfidf)
        lr_pred = predict_sentiment(user_input, lr_model, tfidf)
        svm_pred = predict_sentiment(user_input, svm_model, tfidf)
        rf_pred = predict_sentiment(user_input, rf_model, tfidf)

        c1, c2, c3, c4 = st.columns(4)

        predictions = [
            ("Naive Bayes", nb_pred),
            ("Logistic Regression", lr_pred),
            ("SVM", svm_pred),
            ("Random Forest", rf_pred),
        ]

        for col, (name, pred) in zip([c1, c2, c3, c4], predictions):

            with col:

                css = (
                    "predict-positive"
                    if "POSITIVE" in pred
                    else "predict-negative"
                )

                st.markdown(f"""
                <div class="{css}">
                    <div style="font-size:0.8rem;font-weight:700;">
                        {name}
                    </div>

                    <div class="predict-result">
                        {pred}
                    </div>
                </div>
                """, unsafe_allow_html=True)

st.markdown("### Example Reviews")

examples = [
    ("✅ Positive", "Cepat dapat driver dan perjalanan nyaman."),
    ("❌ Negative", "Aplikasi sering error dan lemot."),
    ("⚠️ Mixed", "Harganya murah tapi aplikasi kadang lag."),
]

cols = st.columns(3)

for i, (title, text) in enumerate(examples):

    with cols[i]:

        st.markdown(f"""
        <div class="card">
            <div style="font-weight:700;margin-bottom:0.5rem;">
                {title}
            </div>

            <div style="font-size:0.9rem;color:#555;">
                {text}
            </div>
        </div>
        """, unsafe_allow_html=True)
