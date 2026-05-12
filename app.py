import streamlit as st
import pandas as pd
import os
import re
import string
import nltk
import pickle
import gdown
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    font-weight: 400;
}
.hero * { color: white !important; }

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

.card {
    background: white;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.3rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid rgba(0,170,91,0.1);
    min-height: 100px;
}

.metric-box {
    background: white;
    border-radius: 12px;
    padding: 1rem 0.8rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border-top: 4px solid var(--green);
    margin-bottom: 0.5rem;
}
.metric-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: #888 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.metric-value {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--green) !important;
}
.metric-value-orange {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.5rem;
    font-weight: 700;
    color: #FF6B35 !important;
}
.metric-value-purple {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.5rem;
    font-weight: 700;
    color: #7C3AED !important;
}
.metric-value-blue {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.5rem;
    font-weight: 700;
    color: #0284C7 !important;
}

.predict-positive {
    background: #E6F7EE;
    border: 2px solid #00AA5B;
    border-radius: 16px;
    padding: 1rem 1.5rem;
    text-align: center;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #00774A !important;
    margin-bottom: 0.5rem;
}
.predict-negative {
    background: #FFF0ED;
    border: 2px solid #FF6B35;
    border-radius: 16px;
    padding: 1rem 1.5rem;
    text-align: center;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #C94A1A !important;
    margin-bottom: 0.5rem;
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

.section-title {
    font-family: 'Space Mono', monospace !important;
    font-size: 1rem;
    font-weight: 700;
    color: #1A1A2E !important;
    border-left: 4px solid var(--green);
    padding-left: 0.75rem;
    margin-bottom: 1rem;
}

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
.stButton > button { height: 45px !important; }

[data-baseweb="popover"] li,
[data-baseweb="menu"] li,
[role="option"] {
    background: white !important;
    color: #1A1A2E !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[role="option"]:hover { background: #E6F7EE !important; }

.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }

#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── NLTK ─────────────────────────────────────────────────────────────────────
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

# ── Preprocessing ─────────────────────────────────────────────────────────────
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
    cleaned   = clean_text(text)
    processed = preprocess_text(cleaned)
    vect      = tfidf.transform([processed])
    pred      = model.predict(vect)[0]
    return "positive" if pred == 1 else "negative"

# ── Load models + compute full metrics from datasets ──────────────────────────
@st.cache_resource
def load_models():
    try:
        # ── pkl DS1 ──
        path_pkl1 = "models_ds1.pkl"
        if not os.path.exists(path_pkl1):
            gdown.download("https://drive.google.com/uc?id=1zETgqcVks_f05gFAckpKVm1TLrFEJ2k3", path_pkl1, quiet=False)
        with open(path_pkl1, 'rb') as f:
            m1 = pickle.load(f)

        # ── pkl DS2 ──
        path_pkl2 = "models_ds2.pkl"
        if not os.path.exists(path_pkl2):
            gdown.download("https://drive.google.com/uc?id=1zPVTqSYUSk22Bq9XxfNNUE8Qp_n_lLD-", path_pkl2, quiet=False)
        with open(path_pkl2, 'rb') as f:
            m2 = pickle.load(f)

        # ── csv DS1 ──
        path_csv1 = "dataset1_gojek.csv"
        if not os.path.exists(path_csv1):
            gdown.download("https://drive.google.com/uc?id=1zWjvbR4WBEMZ7Gipz9nbQ26cSLuW8hi3", path_csv1, quiet=False)
        df1 = pd.read_csv(path_csv1).dropna().drop_duplicates()
        df1['clean_review']     = df1['review'].apply(clean_text)
        df1['processed_review'] = df1['clean_review'].apply(preprocess_text)
        df1['label']            = df1['rate'].map({'positive': 1, 'negative': 0})

        # ── csv DS2 ──
        path_csv2 = "dataset2_gojek.csv"
        if not os.path.exists(path_csv2):
            gdown.download("https://drive.google.com/uc?id=1URNmAxxjCzuYvRDnl6FLOtNbjZ9uIS2R", path_csv2, quiet=False)
        df2 = pd.read_csv(path_csv2)
        df2.columns            = df2.columns.str.strip()
        df2                    = df2[df2['score'] != 3]
        df2['label']           = df2['score'].apply(lambda x: 0 if x <= 2 else 1)
        df2['clean_review']    = df2['content'].apply(clean_text)
        df2['processed_review']= df2['clean_review'].apply(preprocess_text)

        def compute_metrics(df, m):
            _, X_test, _, y_test = train_test_split(
                df['processed_review'], df['label'],
                test_size=0.2, random_state=42, stratify=df['label']
            )
            X_tfidf = m['tfidf'].transform(X_test)
            out = {}
            for key in ['nb', 'lr', 'svm', 'rf']:
                y_pred = m[key].predict(X_tfidf)
                out[key] = {
                    'acc':  accuracy_score(y_test, y_pred),
                    'prec': precision_score(y_test, y_pred, zero_division=0),
                    'rec':  recall_score(y_test, y_pred, zero_division=0),
                    'f1':   f1_score(y_test, y_pred, zero_division=0),
                }
            return out

        results = {
            "Dataset 1": {
                'tfidf':   m1['tfidf'], 'nb': m1['nb'], 'lr': m1['lr'],
                'svm':     m1['svm'],   'rf': m1['rf'],
                'metrics': compute_metrics(df1, m1),
                'df':      df1,
            },
            "Dataset 2": {
                'tfidf':   m2['tfidf'], 'nb': m2['nb'], 'lr': m2['lr'],
                'svm':     m2['svm'],   'rf': m2['rf'],
                'metrics': compute_metrics(df2, m2),
                'df':      df2,
            },
        }
        return results, None

    except Exception as e:
        return None, str(e)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Sentiment Analyzer ✨</h1>
    <p>Menganalisis ulasan pengguna untuk memahami sentimen publik</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Memuat model..."):
    results, error = load_models()

if error:
    st.error(f"Gagal memuat model: {error}")
    st.stop()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🪄 Prediksi  ", "📊 Performa Model"])

# ── TAB 1 ────────────────────────────────────────────────────────────────────
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
        r        = results[dataset_choice]
        nb_pred  = predict_sentiment(user_input, r['nb'],  r['tfidf'])
        lr_pred  = predict_sentiment(user_input, r['lr'],  r['tfidf'])
        svm_pred = predict_sentiment(user_input, r['svm'], r['tfidf'])
        rf_pred  = predict_sentiment(user_input, r['rf'],  r['tfidf'])

        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        for col, label, pred in [
            (c1, "Naive Bayes",         nb_pred),
            (c2, "Logistic Regression", lr_pred),
            (c3, "SVM",                 svm_pred),
            (c4, "Random Forest",       rf_pred),
        ]:
            css  = "predict-positive" if pred == "positive" else "predict-negative"
            icon = "✅" if pred == "positive" else "❌"
            with col:
                st.markdown(f'''<div class="{css}">
                    <div class="predict-label">{icon} {label}</div>
                    <div class="predict-result">{pred.upper()}</div>
                </div>''', unsafe_allow_html=True)

    elif predict_btn:
        st.warning("Masukkan review terlebih dahulu!")

    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Contoh Review</div>', unsafe_allow_html=True)
    examples = [
        ("✅ Positif", "Cepat dapat driver dan perjalanan lancar, sejauh ini puas pakainya!"),
        ("❌ Negatif", "Aplikasi sering error, driver tidak mau jemput, sangat mengecewakan"),
        ("⚠️ Mixed",  "Harga terjangkau tapi aplikasi lambat dan sering lag"),
    ]
    cols = st.columns(3)
    for i, (label, text) in enumerate(examples):
        with cols[i]:
            st.markdown(f"""
            <div class="card" style="display:flex;flex-direction:column;justify-content:flex-start;">
                <div style="font-weight:700;font-size:0.8rem;margin-bottom:0.4rem;color:#1A1A2E;">{label}</div>
                <div style="font-size:0.75rem;color:#555;line-height:1.4;">{text}</div>
            </div>
            """, unsafe_allow_html=True)

# ── TAB 2 ────────────────────────────────────────────────────────────────────
with tab2:
    ds_select = st.radio("Pilih Dataset:", ["Dataset 1", "Dataset 2"], horizontal=True)
    r       = results[ds_select]
    metrics = r['metrics']

    model_configs = [
        ("Naive Bayes",         "nb",  "#00AA5B", "metric-value"),
        ("Logistic Regression", "lr",  "#FF6B35", "metric-value-orange"),
        ("SVM",                 "svm", "#7C3AED", "metric-value-purple"),
        ("Random Forest",       "rf",  "#0284C7", "metric-value-blue"),
    ]

    metric_keys   = [("acc", "Accuracy"), ("prec", "Precision"), ("rec", "Recall"), ("f1", "F1-Score")]

    for model_name, model_key, color, css_class in model_configs:
        st.markdown(
            f'<div class="section-title" style="margin-top:1.2rem;border-left-color:{color};">{model_name}</div>',
            unsafe_allow_html=True
        )
        cols = st.columns(4)
        for col, (mk, ml) in zip(cols, metric_keys):
            val = metrics[model_key][mk]
            with col:
                st.markdown(f'''
                <div class="metric-box" style="border-top-color:{color};">
                    <div class="metric-label">{ml}</div>
                    <div class="{css_class}">{val:.2%}</div>
                </div>''', unsafe_allow_html=True)
