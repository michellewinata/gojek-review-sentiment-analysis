import streamlit as st
import pandas as pd
import os
import re
import string
import nltk
import numpy as np
import gdown
import pickle

from nltk.corpus import stopwords

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="✨",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');

:root {
    --green: #00AA5B;
    --dark: #1A1A2E;
    --accent: #FF6B35;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: #F0F4F8;
}

p, span, label, li, a,
.stMarkdown p, [data-testid="stWidgetLabel"] p,
.stRadio label, .stSelectbox label, .stTextArea label {
    color: #1A1A2E !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
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
    margin: 0 0 0.5rem 0;
    letter-spacing: -1px;
}

.hero p {
    color: rgba(255,255,255,0.9) !important;
    margin: 0;
    font-size: 1.1rem;
}

.hero * {
    color: white !important;
}

.stTextArea textarea {
    background: white !important;
    color: #1A1A2E !important;
    border-radius: 10px !important;
    border: 1.5px solid #ddd !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stSelectbox > div > div,
.stSelectbox > div > div > div,
[data-baseweb="select"] span {
    background: white !important;
    color: #1A1A2E !important;
}

.card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 0.3rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid rgba(0,170,91,0.1);
}

.metric-box {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #888 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.8rem;
    font-weight: 700;
}

.metric-value-green {
    color: #00AA5B !important;
}

.metric-value-orange {
    color: #FF6B35 !important;
}

.metric-value-purple {
    color: #7C3AED !important;
}

.metric-value-blue {
    color: #2563EB !important;
}

.predict-positive {
    background: #E6F7EE;
    border: 2px solid #00AA5B;
    border-radius: 16px;
    padding: 1rem 1.5rem;
    text-align: center;
    color: #00774A !important;
}

.predict-negative {
    background: #FFF0ED;
    border: 2px solid #FF6B35;
    border-radius: 16px;
    padding: 1rem 1.5rem;
    text-align: center;
    color: #C94A1A !important;
}

.predict-label {
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
}

.predict-result {
    font-size: 1.3rem;
    font-weight: 800;
}

.section-title {
    font-family: 'Space Mono', monospace !important;
    font-size: 1rem;
    font-weight: 700;
    color: #1A1A2E !important;
    border-left: 4px solid var(--green);
    padding-left: 0.75rem;
    margin-bottom: 1rem;
}

.stButton > button {
    background: var(--green) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    height: 45px !important;
}

#MainMenu, footer, header {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

setup_nltk()

important_words = {
    'tidak', 'tak', 'bukan', 'belum', 'jangan',
    'kurang', 'gagal', 'susah', 'buruk',
    'jelek', 'parah', 'lambat', 'lemot'
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

def predict_sentiment(text, model, tfidf):
    cleaned = clean_text(text)
    processed = preprocess_text(cleaned)
    vect = tfidf.transform([processed])
    pred = model.predict(vect)[0]
    return "positive" if pred == 1 else "negative"

@st.cache_data
def load_models():
    try:
        model1_id = "1f0OBJGZNhmIUMMqBNOYQOexZrh7XUSkp"
        model2_id = "1fo2n7JHmS8WavNCbOAyUSVlIu4bo3wHF"

        model1_path = "models_ds1.pkl"
        model2_path = "models_ds2.pkl"

        if not os.path.exists(model1_path):
            gdown.download(
                f"https://drive.google.com/uc?id={model1_id}",
                model1_path,
                quiet=False
            )

        if not os.path.exists(model2_path):
            gdown.download(
                f"https://drive.google.com/uc?id={model2_id}",
                model2_path,
                quiet=False
            )

        with open(model1_path, "rb") as f:
            ds1 = pickle.load(f)

        with open(model2_path, "rb") as f:
            ds2 = pickle.load(f)

        results = {
            "Dataset 1": ds1,
            "Dataset 2": ds2
        }

        return results, None

    except Exception as e:
        return None, str(e)

st.markdown("""
<div class="hero">
    <h1>Sentiment Analyzer ✨</h1>
    <p>Menganalisis ulasan pengguna untuk memahami sentimen publik</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading models..."):
    results, error = load_models()

if error:
    st.error(error)
    st.stop()

tab1, tab2 = st.tabs([
    "🪄 Prediksi",
    "📊 Performa Model & Dataset"
])

with tab1:

    st.markdown(
        '<div class="section-title">Coba Analisis Sentimen</div>',
        unsafe_allow_html=True
    )

    user_input = st.text_area(
        "Masukkan ulasan:",
        placeholder="Contoh: Aplikasinya enak dipakai, order cepat dan drivernya ramah",
        height=120
    )

    col_ds, col_btn = st.columns([2,1])

    with col_ds:
        dataset_choice = st.selectbox(
            "Dataset yang digunakan:",
            ["Dataset 1", "Dataset 2"]
        )

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button(
            "Analisis Sentimen →",
            use_container_width=True
        )

    if predict_btn and user_input.strip():

        r = results[dataset_choice]

        nb_pred = predict_sentiment(user_input, r['nb'], r['tfidf'])
        lr_pred = predict_sentiment(user_input, r['lr'], r['tfidf'])
        svm_pred = predict_sentiment(user_input, r['svm'], r['tfidf'])
        rf_pred = predict_sentiment(user_input, r['rf'], r['tfidf'])

        c1, c2, c3, c4 = st.columns(4)

        for col, pred, model_name in zip(
            [c1, c2, c3, c4],
            [nb_pred, lr_pred, svm_pred, rf_pred],
            ["Naive Bayes", "Logistic Regression", "SVM", "Random Forest"]
        ):
            with col:

                css = (
                    "predict-positive"
                    if pred == "positive"
                    else "predict-negative"
                )

                icon = (
                    "✅"
                    if pred == "positive"
                    else "❌"
                )

                st.markdown(f"""
                <div class="{css}">
                    <div class="predict-label">
                        {icon} {model_name}
                    </div>
                    <div class="predict-result">
                        {pred.upper()}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    elif predict_btn:
        st.warning("Masukkan review terlebih dahulu!")

with tab2:

    ds_select = st.radio(
        "Pilih Dataset:",
        ["Dataset 1", "Dataset 2"],
        horizontal=True
    )

    r = results[ds_select]

    models = [
        ("Naive Bayes", "nb", "metric-value-green", "#00AA5B"),
        ("Logistic Regression", "lr", "metric-value-orange", "#FF6B35"),
        ("SVM", "svm", "metric-value-purple", "#7C3AED"),
        ("Random Forest", "rf", "metric-value-blue", "#2563EB")
    ]

    for title, key, color_class, border_color in models:

        st.markdown(
            f'''
            <div class="section-title"
            style="margin-top:1.5rem;border-left-color:{border_color};">
            {title}
            </div>
            ''',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        m = r['metrics'][key]

        for col, label, val in zip(
            [c1, c2, c3, c4],
            ["Accuracy", "Precision", "Recall", "F1-Score"],
            [m['acc'], m['prec'], m['rec'], m['f1']]
        ):

            with col:

                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">
                        {label}
                    </div>
                    <div class="metric-value {color_class}">
                        {val:.2%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
