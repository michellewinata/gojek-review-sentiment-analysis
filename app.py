import streamlit as st
import pandas as pd
import re
import string
import nltk
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import io
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

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

.stApp { background: #F0F4F8; }

/* Force semua teks jadi hitam */
p, span, label, li, a,
.stMarkdown p, [data-testid="stWidgetLabel"] p,
.stRadio label, .stSelectbox label, .stTextArea label {
    color: #1A1A2E !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Hero */
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
    font-weight: 400;
}
            
.hero * {
    color: white !important;
}

/* Input */
.stTextArea textarea {
    background: white !important;
    color: #1A1A2E !important;
    border-radius: 10px !important;
    border: 1.5px solid #ddd !important;
    box-shadow: none !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stTextArea textarea::placeholder {
    color: #aaa !important;
    opacity: 1 !important;
}
.stSelectbox > div > div,
.stSelectbox > div > div > div,
[data-baseweb="select"] span {
    background: white !important;
    color: #1A1A2E !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Card */
.card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 0.3rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid rgba(0,170,91,0.1);
    min-height: 140px;
}

/* Metric */
.metric-box {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border-top: 4px solid var(--green);
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #888 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
            
.metric-value {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--green) !important;
}
            
.metric-value-orange {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.8rem;
    font-weight: 700;
    color: #FF6B35 !important;
}

/* Predict box */
.predict-positive {
    background: #E6F7EE;
    border: 2px solid #00AA5B;
    border-radius: 16px;
    padding: 1rem 1.5rem;
    text-align: center;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #00774A !important;
}
.predict-negative {
    background: #FFF0ED;
    border: 2px solid #FF6B35;
    border-radius: 16px;
    padding: 1rem 1.5rem;
    text-align: center;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #C94A1A !important;
}
.predict-label {
    font-size: 0.8rem;
    font-weight: 600;
    opacity: 0.7;
    margin-bottom: 0.2rem;
}
.predict-result {
    font-size: 1.3rem;
    font-weight: 800;
}

/* Section title */
.section-title {
    font-family: 'Space Mono', monospace !important;
    font-size: 1rem;
    font-weight: 700;
    color: #1A1A2E !important;
    border-left: 4px solid var(--green);
    padding-left: 0.75rem;
    margin-bottom: 1rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: white;
    border-radius: 12px;
    padding: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    color: #1A1A2E !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Button */
.stButton > button, .stButton > button * {
    background: var(--green) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #007A40 !important;
    transform: translateY(-1px) !important;
}

.stButton > button {
    height: 45px !important;
}
            
[data-baseweb="popover"] li,
[data-baseweb="menu"] li,
[role="option"] {
    background: white !important;
    color: #1A1A2E !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[role="option"]:hover {
    background: #E6F7EE !important;
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1rem;
}
            
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Code
@st.cache_resource
def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

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

def predict_sentiment(text, model, tfidf):
    cleaned = clean_text(text)
    processed = preprocess_text(cleaned)
    vect = tfidf.transform([processed])
    pred = model.predict(vect)[0]
    return "positive" if pred == 1 else "negative"

def download_from_gdrive(file_id, dest_path):
    session = requests.Session()
    url = "https://drive.google.com/uc?export=download"
    
    # Step 1: ambil confirm token
    response = session.get(url, params={"id": file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    
    # Step 2: download dengan token kalau ada
    if token:
        response = session.get(url, params={"id": file_id, "confirm": token}, stream=True)
    
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

@st.cache_data
def load_and_train():
    try:
        path1 = "/tmp/dataset1_gojek.csv"
        path2 = "/tmp/dataset2_gojek.csv"

        if not os.path.exists(path1):
            download_from_gdrive("1zWjvbR4WBEMZ7Gipz9nbQ26cSLuW8hi3", path1)
        if not os.path.exists(path2):
            download_from_gdrive("1URNmAxxjCzuYvRDnl6FLOtNbjZ9uIS2R", path2)

        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)

        results = {}
        for name, df, text_col in [("Dataset 1", df1, 'processed_review'), ("Dataset 2", df2, 'processed_review')]:
            X = df[text_col]
            y = df['label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
            X_train_tfidf = tfidf.fit_transform(X_train)
            X_test_tfidf = tfidf.transform(X_test)

            ratio = y.value_counts(normalize=True).sort_index().values
            nb = MultinomialNB(class_prior=ratio)
            nb.fit(X_train_tfidf, y_train)
            y_pred_nb = nb.predict(X_test_tfidf)

            lr = LogisticRegression(max_iter=1000, class_weight='balanced')
            lr.fit(X_train_tfidf, y_train)
            y_pred_lr = lr.predict(X_test_tfidf)

            results[name] = {
                'tfidf': tfidf, 'nb': nb, 'lr': lr,
                'y_test': y_test, 'y_pred_nb': y_pred_nb, 'y_pred_lr': y_pred_lr,
                'df': df,
                'metrics': {
                    'nb': {
                        'acc': accuracy_score(y_test, y_pred_nb),
                        'prec': precision_score(y_test, y_pred_nb, zero_division=0),
                        'rec': recall_score(y_test, y_pred_nb, zero_division=0),
                        'f1': f1_score(y_test, y_pred_nb, zero_division=0),
                    },
                    'lr': {
                        'acc': accuracy_score(y_test, y_pred_lr),
                        'prec': precision_score(y_test, y_pred_lr, zero_division=0),
                        'rec': recall_score(y_test, y_pred_lr, zero_division=0),
                        'f1': f1_score(y_test, y_pred_lr, zero_division=0),
                    }
                }
            }

        return results, None

    except Exception as e:
        return None, str(e)
        
# HERO 
st.markdown("""
<div class="hero">
    <h1>Sentiment Analyzer ✨</h1>
    <p>Menganalisis ulasan pengguna untuk memahami sentimen publik</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Training models..."):
    results, error = load_and_train()

if error:
    st.error("Error loading datasets!")
    st.stop()

# TABS 
tab1, tab2 = st.tabs(["🪄 Prediksi  ", "📊 Performa Model & Dataset"])

# TAB 1
with tab1:
    st.markdown('<div class="section-title">Coba Analisis Sentimen</div>', unsafe_allow_html=True)

    user_input = st.text_area(
        "Masukkan ulasan:",
        placeholder="Contoh: Aplikasinya enak dipakai, order cepat dan drivernya ramah",
        height=120
    )

    col_ds, col_btn = st.columns([2, 1])
    with col_ds:
        dataset_choice = st.selectbox("Dataset yang digunakan:", ["Dataset 1", "Dataset 2"])
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Analisis Sentimen →", use_container_width=True)

    if predict_btn and user_input.strip():
        r = results[dataset_choice]
        nb_pred = predict_sentiment(user_input, r['nb'], r['tfidf'])
        lr_pred = predict_sentiment(user_input, r['lr'], r['tfidf'])
        c1, c2 = st.columns(2)
        with c1:
            css = "predict-positive" if nb_pred == "positive" else "predict-negative"
            icon = "✅" if nb_pred == "positive" else "❌"
            st.markdown(f'''<div class="{css}">
                <div class="predict-label">{icon} Naive Bayes</div>
                <div class="predict-result">{nb_pred.upper()}</div>
            </div>''', unsafe_allow_html=True)
        with c2:
            css = "predict-positive" if lr_pred == "positive" else "predict-negative"
            icon = "✅" if lr_pred == "positive" else "❌"
            st.markdown(f'''<div class="{css}">
                <div class="predict-label">{icon} Logistic Regression</div>
                <div class="predict-result">{lr_pred.upper()}</div>
            </div>''', unsafe_allow_html=True)
    elif predict_btn:
        st.warning("Masukkan review terlebih dahulu!")

    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Contoh Review</div>', unsafe_allow_html=True)
    examples = [
        ("✅ Positif", "Cepat dapat driver dan perjalanan lancar, sejauh ini puas pakainya!"),
        ("❌ Negatif", "Aplikasi sering error, driver tidak mau jemput, sangat mengecewakan"),
        ("⚠️ Mixed", "Harga terjangkau tapi aplikasi lambat dan sering lag"),
    ]

    cols = st.columns(3)
    for i, (label, text) in enumerate(examples):
        with cols[i]:
            st.markdown(f"""
            <div class="card" style="height:140px;display:flex;flex-direction:column;justify-content:flex-start;">
                <div style="font-weight:700;margin-bottom:0.5rem;color:#1A1A2E;">{label}</div>
                <div style="font-size:0.85rem;color:#555;">{text}</div>
            </div>
            """, unsafe_allow_html=True)

# TAB 2
with tab2:
    col1, col2 = st.columns(2)
    for col, ds_name in zip([col1, col2], ["Dataset 1", "Dataset 2"]):
        with col:
            df = results[ds_name]['df']
            total = len(df)
            pos = (df['label'] == 1).sum()
            neg = (df['label'] == 0).sum()
            st.markdown(f"""
            <div class="card">
                <div style="font-weight:700;font-size:1rem;margin-bottom:1rem;color:#1A1A2E;">{ds_name}</div>
                <div style="display:flex;gap:1rem;justify-content:space-around;text-align:center;">
                    <div>
                        <div style="font-size:1.5rem;font-weight:800;color:#1A1A2E;">{total:,}</div>
                        <div style="color:#888;font-size:0.75rem;">Total Reviews</div>
                    </div>
                    <div>
                        <div style="font-size:1.5rem;font-weight:800;color:#00AA5B;">{pos:,}</div>
                        <div style="color:#888;font-size:0.75rem;">Positive</div>
                    </div>
                    <div>
                        <div style="font-size:1.5rem;font-weight:800;color:#FF6B35;">{neg:,}</div>
                        <div style="color:#888;font-size:0.75rem;">Negative</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr style="margin-top:-1rem;margin-bottom:1rem;border:none;border-top:1px solid #ddd;">', unsafe_allow_html=True)

    ds_select = st.radio("Pilih Dataset:", ["Dataset 1", "Dataset 2"], horizontal=True)
    r = results[ds_select]

    st.markdown('<div class="section-title">Naive Bayes</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    m = r['metrics']['nb']
    for col, label, val in zip([c1,c2,c3,c4], ["Accuracy","Precision","Recall","F1-Score"], [m['acc'],m['prec'],m['rec'],m['f1']]):
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-label">{label}</div><div class="metric-value">{val:.2%}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:1.5rem;border-left-color:#FF6B35;">Logistic Regression</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    m = r['metrics']['lr']
    for col, label, val in zip([c1,c2,c3,c4], ["Accuracy","Precision","Recall","F1-Score"], [m['acc'],m['prec'],m['rec'],m['f1']]):
        with col:
            st.markdown(f'<div class="metric-box" style="border-top-color:#FF6B35;"><div class="metric-label">{label}</div><div class="metric-value-orange">{val:.2%}</div></div>', unsafe_allow_html=True)
