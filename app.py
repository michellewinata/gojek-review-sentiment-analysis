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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
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
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 0.3rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid rgba(0,170,91,0.1);
    min-height: 140px;
}

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
.metric-value-purple {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.8rem;
    font-weight: 700;
    color: #7C3AED !important;
}
.metric-value-blue {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.8rem;
    font-weight: 700;
    color: #2563EB !important;
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

# ── NLTK ──────────────────────────────────────────────────────────────────────
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

# ── TEXT UTILS ────────────────────────────────────────────────────────────────
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

# ── LOAD / TRAIN ──────────────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    """
    Tries to load pre-trained models from Google Drive (.pkl).
    If the pkl contains a full bundle (tfidf + models + metrics), uses them.
    Otherwise falls back to training from CSV datasets.
    """
    try:
        # ── Drive model files ──────────────────────────────────────────────
        model_configs = {
            "Dataset 1": {
                "model_id":   "1f0OBJGZNhmIUMMqBNOYQOexZrh7XUSkp",
                "model_path": "model1_gojek.pkl",
                "csv_id":     "1zWjvbR4WBEMZ7Gipz9nbQ26cSLuW8hi3",
                "csv_path":   "dataset1_gojek.csv",
                "text_col":   "review",
                "label_col":  "rate",
                "label_map":  {"positive": 1, "negative": 0},
            },
            "Dataset 2": {
                "model_id":   "1fo2n7JHmS8WavNCbOAyUSVlIu4bo3wHF",
                "model_path": "model2_gojek.pkl",
                "csv_id":     "1URNmAxxjCzuYvRDnl6FLOtNbjZ9uIS2R",
                "csv_path":   "dataset2_gojek.csv",
                "text_col":   "content",
                "label_col":  "score",
                "label_map":  None,            # numeric → handled below
            },
        }

        results = {}

        for ds_name, cfg in model_configs.items():

            # ── 1. Try loading pre-trained pkl ─────────────────────────────
            pkl_path = cfg["model_path"]
            bundle   = None

            if not os.path.exists(pkl_path):
                url = f"https://drive.google.com/uc?id={cfg['model_id']}"
                gdown.download(url, pkl_path, quiet=False)

            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    bundle = pickle.load(f)

            # ── 2. Load CSV (needed for stats & fallback training) ──────────
            csv_path = cfg["csv_path"]
            if not os.path.exists(csv_path):
                url = f"https://drive.google.com/uc?id={cfg['csv_id']}"
                gdown.download(url, csv_path, quiet=False)

            df = pd.read_csv(csv_path).dropna().drop_duplicates()
            df.columns = df.columns.str.strip()

            if cfg["label_map"]:
                df['label'] = df[cfg["label_col"]].map(cfg["label_map"])
            else:
                df = df[df['score'] != 3]
                df['label'] = df['score'].apply(lambda x: 0 if x <= 2 else 1)

            df['clean_review']     = df[cfg["text_col"]].apply(clean_text)
            df['processed_review'] = df['clean_review'].apply(preprocess_text)
            df = df.dropna(subset=['label', 'processed_review'])

            X = df['processed_review']
            y = df['label'].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # ── 3. Resolve tfidf ───────────────────────────────────────────
            if isinstance(bundle, dict) and 'tfidf' in bundle:
                tfidf = bundle['tfidf']
            else:
                tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
                tfidf.fit(X_train)

            X_train_tfidf = tfidf.transform(X_train)
            X_test_tfidf  = tfidf.transform(X_test)

            # ── 4. Resolve / train each model ──────────────────────────────
            def _get_or_train(key, estimator):
                """Return model from bundle if present, else train estimator."""
                if isinstance(bundle, dict) and key in bundle:
                    return bundle[key]
                estimator.fit(X_train_tfidf, y_train)
                return estimator

            ratio = y.value_counts(normalize=True).sort_index().values

            nb  = _get_or_train('nb',  MultinomialNB(class_prior=ratio))
            lr  = _get_or_train('lr',  LogisticRegression(max_iter=1000, class_weight='balanced'))
            svm = _get_or_train('svm', LinearSVC(max_iter=2000, class_weight='balanced'))
            rf  = _get_or_train('rf',  RandomForestClassifier(
                                            n_estimators=200,
                                            class_weight='balanced',
                                            random_state=42,
                                            n_jobs=-1,
                                        ))

            # ── 5. Predictions ─────────────────────────────────────────────
            y_pred_nb  = nb.predict(X_test_tfidf)
            y_pred_lr  = lr.predict(X_test_tfidf)
            y_pred_svm = svm.predict(X_test_tfidf)
            y_pred_rf  = rf.predict(X_test_tfidf)

            def _metrics(y_pred):
                return {
                    'acc':  accuracy_score(y_test, y_pred),
                    'prec': precision_score(y_test, y_pred, zero_division=0),
                    'rec':  recall_score(y_test, y_pred, zero_division=0),
                    'f1':   f1_score(y_test, y_pred, zero_division=0),
                }

            results[ds_name] = {
                'tfidf': tfidf,
                'nb': nb, 'lr': lr, 'svm': svm, 'rf': rf,
                'y_test': y_test,
                'y_pred_nb':  y_pred_nb,
                'y_pred_lr':  y_pred_lr,
                'y_pred_svm': y_pred_svm,
                'y_pred_rf':  y_pred_rf,
                'df': df,
                'metrics': {
                    'nb':  _metrics(y_pred_nb),
                    'lr':  _metrics(y_pred_lr),
                    'svm': _metrics(y_pred_svm),
                    'rf':  _metrics(y_pred_rf),
                },
            }

        return results, None

    except Exception as e:
        import traceback
        return None, traceback.format_exc()


# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Sentiment Analyzer ✨</h1>
    <p>Menganalisis ulasan pengguna untuk memahami sentimen publik</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Memuat model..."):
    results, error = load_and_train()

if error:
    st.error(error)
    st.stop()

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🪄 Prediksi  ", "📊 Performa Model & Dataset"])

# ── TAB 1: PREDICTION ─────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Coba Analisis Sentimen</div>', unsafe_allow_html=True)

    user_input = st.text_area(
        "Masukkan ulasan:",
        placeholder="Contoh: Aplikasinya enak dipakai, order cepat dan drivernya ramah",
        height=120,
    )

    col_ds, col_btn = st.columns([2, 1])
    with col_ds:
        dataset_choice = st.selectbox("Dataset yang digunakan:", ["Dataset 1", "Dataset 2"])
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Analisis Sentimen →", use_container_width=True)

    if predict_btn and user_input.strip():
        r = results[dataset_choice]

        model_defs = [
            ("Naive Bayes",        r['nb'],  "🟢"),
            ("Logistic Regression",r['lr'],  "🟠"),
            ("SVM",                r['svm'], "🟣"),
            ("Random Forest",      r['rf'],  "🔵"),
        ]

        cols = st.columns(2)
        for idx, (model_name, model_obj, icon) in enumerate(model_defs):
            pred = predict_sentiment(user_input, model_obj, r['tfidf'])
            css  = "predict-positive" if pred == "positive" else "predict-negative"
            tick = "✅" if pred == "positive" else "❌"
            with cols[idx % 2]:
                st.markdown(f'''<div class="{css}">
                    <div class="predict-label">{tick} {model_name}</div>
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
            <div class="card" style="height:140px;display:flex;flex-direction:column;justify-content:flex-start;">
                <div style="font-weight:700;margin-bottom:0.5rem;color:#1A1A2E;">{label}</div>
                <div style="font-size:0.85rem;color:#555;">{text}</div>
            </div>
            """, unsafe_allow_html=True)

# ── TAB 2: METRICS & DATASET ──────────────────────────────────────────────────
with tab2:
    # Dataset summary cards
    col1, col2 = st.columns(2)
    for col, ds_name in zip([col1, col2], ["Dataset 1", "Dataset 2"]):
        with col:
            df  = results[ds_name]['df']
            total = len(df)
            pos   = (df['label'] == 1).sum()
            neg   = (df['label'] == 0).sum()
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

    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    model_sections = [
        ("Naive Bayes",         "nb",  "var(--green)", "metric-value"),
        ("Logistic Regression", "lr",  "#FF6B35",      "metric-value-orange"),
        ("SVM",                 "svm", "#7C3AED",      "metric-value-purple"),
        ("Random Forest",       "rf",  "#2563EB",      "metric-value-blue"),
    ]

    for model_name, key, color, val_class in model_sections:
        st.markdown(
            f'<div class="section-title" style="margin-top:1.5rem;border-left-color:{color};">'
            f'{model_name}</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        m = r['metrics'][key]
        for col, label, val in zip(
            [c1, c2, c3, c4],
            metric_labels,
            [m['acc'], m['prec'], m['rec'], m['f1']],
        ):
            with col:
                st.markdown(
                    f'<div class="metric-box" style="border-top-color:{color};">'
                    f'<div class="metric-label">{label}</div>'
                    f'<div class="{val_class}">{val:.2%}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
