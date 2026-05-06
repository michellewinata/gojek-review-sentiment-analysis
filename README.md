📌 Deskripsi
Proyek ini menganalisis sentimen ulasan pengguna aplikasi Gojek menggunakan Natural Language Processing (NLP). Fokus utama proyek adalah membandingkan performa model pada dua dataset dengan karakteristik berbeda (dataset kecil berlabel vs dataset besar tanpa label).

🎯 Tujuan
- Klasifikasi sentimen (positif & negatif)
- Membandingkan performa model
- Melihat pengaruh ukuran & labeling data

📊 Dataset
Dataset 1: ±1.780 data (sudah berlabel)
     link: https://www.kaggle.com/datasets/yundarastaandini/gojek-review
Dataset 2: ±210.000 data (label dari rating)
     link: https://www.kaggle.com/datasets/ucupsedaya/gojek-app-reviews-bahasa-indonesia

🚀 Cara Menjalankan
git clone https://github.com/username/gojek-sentiment-analysis.git
cd gojek-sentiment-analysis
pip install -r requirements.txt
streamlit run app.py
