import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import re, html, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from wordcloud import WordCloud

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ── CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Sentiment Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── THEME ─────────────────────────────────────────────────
MAROON = "#800000"
GOLD   = "#c9a84c"
GRAY   = "#6b6b6b"
COLORS = {'Positive': '#27ae60', 'Neutral': '#e67e22', 'Negative': '#c0392b'}

st.markdown(f"""
<style>
html, body, [class*="css"] {{ font-family: 'Segoe UI', sans-serif; }}
section[data-testid="stSidebar"] {{ background-color: {MAROON}; }}
section[data-testid="stSidebar"] * {{ color: white !important; }}
.page-header {{
    background: linear-gradient(135deg, {MAROON} 0%, #4a0000 100%);
    padding: 28px 32px 22px 32px; border-radius: 10px;
    color: white; margin-bottom: 24px;
}}
.page-header h1 {{ margin: 0; font-size: 1.8rem; font-weight: 700; }}
.page-header p  {{ margin: 6px 0 0 0; font-size: 0.95rem; opacity: 0.85; }}
.metric-row {{ display: flex; gap: 16px; margin-bottom: 24px; }}
.metric-card {{
    flex: 1; background: white;
    border-left: 5px solid {MAROON}; border-radius: 8px;
    padding: 18px 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}}
.metric-card .val {{ font-size: 1.9rem; font-weight: 700; color: {MAROON}; line-height: 1.1; }}
.metric-card .lbl {{ font-size: 0.82rem; color: {GRAY}; margin-top: 4px;
                     text-transform: uppercase; letter-spacing: 0.5px; }}
.section-title {{
    font-size: 1.05rem; font-weight: 700; color: {MAROON};
    border-bottom: 2px solid {MAROON}; padding-bottom: 6px;
    margin: 24px 0 16px 0; text-transform: uppercase; letter-spacing: 0.4px;
}}
.result-positive {{ background:#eaf7ef; border-left:6px solid #27ae60; border-radius:8px; padding:18px 22px; margin:16px 0; }}
.result-negative {{ background:#fdecea; border-left:6px solid #c0392b; border-radius:8px; padding:18px 22px; margin:16px 0; }}
.result-neutral  {{ background:#fef9ec; border-left:6px solid #e67e22; border-radius:8px; padding:18px 22px; margin:16px 0; }}
.result-label {{ font-size:1.4rem; font-weight:700; margin:0; }}
.result-sub   {{ font-size:0.88rem; color:{GRAY}; margin-top:4px; }}
</style>
""", unsafe_allow_html=True)

# ── UTILS ─────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = html.unescape(str(text)).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

# ── LOAD & TRAIN ──────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model, harap tunggu...")
def load_and_train():
    # Load sample yang sudah bersih dari Colab
    df = pd.read_csv('reviews_sample.csv')

    # Pastikan kolom clean_text ada
    if 'clean_text' not in df.columns:
        df['clean_text'] = df['Text'].apply(preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['Sentiment'],
        test_size=0.2, stratify=df['Sentiment'], random_state=42
    )

    vecs = {
        'DT + BoW':    CountVectorizer(max_features=10000),
        'DT + N-Gram': CountVectorizer(max_features=10000, ngram_range=(2, 2)),
        'DT + TF-IDF': TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    }
    models, reports = {}, {}
    for name, vec in vecs.items():
        m = DecisionTreeClassifier(random_state=42)
        m.fit(vec.fit_transform(X_train), y_train)
        y_pred = m.predict(vec.transform(X_test))
        models[name] = m
        reports[name] = classification_report(y_test, y_pred, output_dict=True)

    best_pred = models['DT + TF-IDF'].predict(vecs['DT + TF-IDF'].transform(X_test))
    cm = confusion_matrix(y_test, best_pred, labels=['Positive', 'Neutral', 'Negative'])

    return models, vecs, reports, cm, df

models, vecs, reports, cm, df_s = load_and_train()

# ── SESSION STATE ─────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── SIDEBAR ───────────────────────────────────────────────
st.sidebar.markdown("## NLP Sentiment Dashboard")
st.sidebar.markdown("Amazon Fine Food Reviews")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigasi", [
    "Overview",
    "Eksplorasi Data",
    "Performa Model",
    "Prediksi Sentimen",
    "Batch Prediksi",
    "Tentang Proyek"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Terbaik:** DT + TF-IDF")
st.sidebar.markdown("**Accuracy:** 77.42%  |  **F1:** 0.7722")

# ════════════════════════════════════════════════════════
# HALAMAN 1 — OVERVIEW
# ════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("""
    <div class="page-header">
        <h1>Sentiment Classification Dashboard</h1>
        <p>Amazon Fine Food Reviews — Pemrosesan Bahasa Alami | Universitas Muhammadiyah Malang 2026</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="metric-row">
        <div class="metric-card"><div class="val">568.454</div><div class="lbl">Total Ulasan</div></div>
        <div class="metric-card"><div class="val">50.000</div><div class="lbl">Sample Digunakan</div></div>
        <div class="metric-card"><div class="val">77.42%</div><div class="lbl">Akurasi Terbaik</div></div>
        <div class="metric-card"><div class="val">0.7722</div><div class="lbl">F1-Score Terbaik</div></div>
        <div class="metric-card"><div class="val">3</div><div class="lbl">Eksperimen Model</div></div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown('<div class="section-title">Tentang Project</div>', unsafe_allow_html=True)
        st.markdown("""
        Dashboard ini merupakan hasil **Mini Project Mata Kuliah Pemrosesan Bahasa Alami (NLP)**
        yang membangun sistem klasifikasi sentimen otomatis pada ulasan produk pangan dari platform Amazon.

        Sistem menggunakan algoritma **Decision Tree** dengan tiga pendekatan ekstraksi fitur
        *(Bag of Words, N-Gram, TF-IDF)* untuk menentukan sentimen ulasan secara otomatis.
        """)

        st.markdown('<div class="section-title">Pipeline</div>', unsafe_allow_html=True)
        pipeline = pd.DataFrame({
            'Tahap': ['1. Data Loading', '2. Data Cleaning', '3. Text Preprocessing',
                      '4. Feature Extraction', '5. Model Training', '6. Evaluasi & Prediksi'],
            'Deskripsi': [
                'Load dataset Reviews.csv (568.454 baris)',
                'Drop missing values, duplikat, teks pendek',
                'Lowercasing, remove stopwords, lemmatization',
                'BoW / N-Gram / TF-IDF Vectorizer',
                'Decision Tree Classifier (Gini, 80:20 split)',
                'Accuracy, Precision, Recall, F1-Score'
            ]
        })
        st.dataframe(pipeline, use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="section-title">Distribusi Kelas Label</div>', unsafe_allow_html=True)
        sent = df_s['Sentiment'].value_counts().reset_index()
        sent.columns = ['Sentimen', 'Jumlah']
        fig = px.pie(sent, names='Sentimen', values='Jumlah',
                     color='Sentimen', color_discrete_map=COLORS, hole=0.4)
        fig.update_layout(margin=dict(t=20,b=20,l=20,r=20), height=280,
                          legend=dict(orientation='h', y=-0.15))
        fig.update_traces(textinfo='percent+label', textfont_size=13)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">10 Data Pertama</div>', unsafe_allow_html=True)
    st.dataframe(df_s[['Score','Sentiment','Summary','Text']].head(10),
                 use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════
# HALAMAN 2 — EKSPLORASI DATA
# ════════════════════════════════════════════════════════
elif page == "Eksplorasi Data":
    st.markdown("""
    <div class="page-header">
        <h1>Eksplorasi Data</h1>
        <p>Analisis distribusi dan karakteristik dataset Amazon Fine Food Reviews</p>
    </div>""", unsafe_allow_html=True)

    filter_sent = st.multiselect("Filter kelas sentimen:",
                                  ['Positive', 'Neutral', 'Negative'],
                                  default=['Positive', 'Neutral', 'Negative'])
    df_f = df_s[df_s['Sentiment'].isin(filter_sent)]
    st.caption(f"Menampilkan {len(df_f):,} dari {len(df_s):,} data")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Distribusi Kelas Sentimen</div>', unsafe_allow_html=True)
        sc = df_f['Sentiment'].value_counts().reset_index()
        sc.columns = ['Sentimen', 'Jumlah']
        fig1 = px.bar(sc, x='Sentimen', y='Jumlah', color='Sentimen',
                      color_discrete_map=COLORS, text='Jumlah')
        fig1.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig1.update_layout(showlegend=False, margin=dict(t=10,b=10), height=320)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Distribusi Rating Score (1–5)</div>', unsafe_allow_html=True)
        rc = df_f['Score'].value_counts().sort_index().reset_index()
        rc.columns = ['Score', 'Jumlah']
        fig2 = px.bar(rc, x='Score', y='Jumlah', text='Jumlah',
                      color_discrete_sequence=[MAROON])
        fig2.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig2.update_layout(showlegend=False, margin=dict(t=10,b=10), height=320)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Distribusi Panjang Teks Ulasan</div>', unsafe_allow_html=True)
    df_f2 = df_f.copy()
    df_f2['Jumlah Kata'] = df_f2['Text'].apply(lambda x: len(str(x).split()))
    fig3 = px.histogram(df_f2, x='Jumlah Kata', color='Sentiment',
                        nbins=60, barmode='overlay', opacity=0.75,
                        color_discrete_map=COLORS)
    fig3.update_layout(xaxis_range=[0,300], margin=dict(t=10,b=10), height=300)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-title">Word Cloud per Kelas Sentimen</div>', unsafe_allow_html=True)
    wc1, wc2, wc3 = st.columns(3)
    for col, sentiment, cmap in zip([wc1,wc2,wc3],
                                     ['Positive','Neutral','Negative'],
                                     ['Greens','Oranges','Reds']):
        with col:
            st.caption(sentiment)
            subset = df_s[df_s['Sentiment'] == sentiment]['clean_text']
            txt = ' '.join(subset.sample(min(1000, len(subset)), random_state=42))
            wc = WordCloud(width=400, height=280, background_color='white',
                           colormap=cmap, prefer_horizontal=0.9).generate(txt)
            fig_wc, ax = plt.subplots(figsize=(5, 3.5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout(pad=0)
            st.pyplot(fig_wc)

    st.markdown('<div class="section-title">Tabel Data Interaktif</div>', unsafe_allow_html=True)
    n = st.slider("Jumlah baris:", 10, 100, 20, step=10)
    st.dataframe(df_f[['Score','Sentiment','Summary','Text']].head(n),
                 use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════
# HALAMAN 3 — PERFORMA MODEL
# ════════════════════════════════════════════════════════
elif page == "Performa Model":
    st.markdown("""
    <div class="page-header">
        <h1>Performa Model</h1>
        <p>Perbandingan hasil eksperimen Decision Tree dengan BoW, N-Gram, dan TF-IDF</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Perbandingan Keseluruhan</div>', unsafe_allow_html=True)
    perf = []
    for name, r in reports.items():
        perf.append({
            'Model':     name,
            'Accuracy':  round(r['accuracy'], 4),
            'Precision': round(r['weighted avg']['precision'], 4),
            'Recall':    round(r['weighted avg']['recall'], 4),
            'F1-Score':  round(r['weighted avg']['f1-score'], 4)
        })
    perf_df = pd.DataFrame(perf)
    st.dataframe(perf_df.style.highlight_max(axis=0, color='#f5e6e6'),
                 use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Visualisasi Perbandingan</div>', unsafe_allow_html=True)
        fig_p = px.bar(perf_df, x='Model', y=['Accuracy','F1-Score'],
                       barmode='group',
                       color_discrete_sequence=[MAROON, GOLD])
        fig_p.update_layout(legend=dict(orientation='h', y=1.1),
                            height=320, margin=dict(t=10,b=10))
        st.plotly_chart(fig_p, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Confusion Matrix — DT + TF-IDF</div>', unsafe_allow_html=True)
        labels = ['Positive', 'Neutral', 'Negative']
        fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels,
                           color_continuous_scale=[[0,'#fff5f5'],[1,MAROON]])
        fig_cm.update_layout(height=320, margin=dict(t=10,b=10),
                             xaxis_title='Predicted', yaxis_title='Actual')
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown('<div class="section-title">Detail Per Kelas Sentimen</div>', unsafe_allow_html=True)
    selected = st.selectbox("Pilih model:", list(reports.keys()))
    r = reports[selected]
    pc = []
    for k in ['Positive','Neutral','Negative']:
        if k in r:
            pc.append({'Kelas': k,
                       'Precision': round(r[k]['precision'],4),
                       'Recall':    round(r[k]['recall'],4),
                       'F1-Score':  round(r[k]['f1-score'],4),
                       'Support':   int(r[k]['support'])})
    pc_df = pd.DataFrame(pc)
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.dataframe(pc_df, use_container_width=True, hide_index=True)
    with col2:
        fig_pc = px.bar(pc_df, x='Kelas', y=['Precision','Recall','F1-Score'],
                        barmode='group',
                        color_discrete_sequence=[MAROON, GOLD, '#555555'])
        fig_pc.update_layout(height=280, margin=dict(t=10,b=10),
                             legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig_pc, use_container_width=True)

# ════════════════════════════════════════════════════════
# HALAMAN 4 — PREDIKSI SENTIMEN
# ════════════════════════════════════════════════════════
elif page == "Prediksi Sentimen":
    st.markdown("""
    <div class="page-header">
        <h1>Prediksi Sentimen</h1>
        <p>Masukkan teks ulasan produk pangan untuk diprediksi sentimennya secara otomatis</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        model_choice = st.selectbox("Model yang digunakan:", list(models.keys()), index=2)
        user_input = st.text_area("Teks ulasan:", height=140,
                                   placeholder="Contoh: This product is amazing! Great taste and very fresh.")
        predict_btn = st.button("Prediksi", type="primary", use_container_width=True)

    with col2:
        st.markdown("**Contoh ulasan:**")
        st.markdown("""
        *Positif:*
        > "This product is absolutely amazing. Great taste and very fresh. Highly recommend!"

        *Netral:*
        > "It was okay. Nothing special but not bad either. Decent for the price."

        *Negatif:*
        > "Terrible product. Arrived damaged and tasted awful. Complete waste of money."
        """)

    if predict_btn:
        if not user_input.strip():
            st.warning("Masukkan teks ulasan terlebih dahulu.")
        else:
            clean  = preprocess(user_input)
            vec    = vecs[model_choice]
            m      = models[model_choice]
            pred   = m.predict(vec.transform([clean]))[0]
            proba  = m.predict_proba(vec.transform([clean]))[0]
            classes = m.classes_

            css_map   = {'Positive':'result-positive','Negative':'result-negative','Neutral':'result-neutral'}
            label_map = {'Positive':'Positive — Ulasan bersifat positif',
                         'Negative':'Negative — Ulasan bersifat negatif',
                         'Neutral': 'Neutral — Ulasan bersifat netral'}

            st.markdown(f"""
            <div class="{css_map[pred]}">
                <p class="result-label">{label_map[pred]}</p>
                <p class="result-sub">Model: {model_choice} &nbsp;|&nbsp; Confidence: {max(proba):.1%}</p>
            </div>""", unsafe_allow_html=True)

            proba_df = pd.DataFrame({'Kelas': classes, 'Confidence': proba})
            fig_pr = px.bar(proba_df, x='Kelas', y='Confidence',
                            color='Kelas', color_discrete_map=COLORS,
                            range_y=[0,1], text='Confidence')
            fig_pr.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig_pr.update_layout(showlegend=False, height=280, margin=dict(t=10,b=10))
            st.plotly_chart(fig_pr, use_container_width=True)

            with st.expander("Lihat teks setelah preprocessing"):
                st.code(clean)

            st.session_state.history.append({
                'Model':       model_choice,
                'Input':       user_input[:70]+'...' if len(user_input)>70 else user_input,
                'Prediksi':    pred,
                'Confidence':  f"{max(proba):.1%}"
            })

    if st.session_state.history:
        st.markdown('<div class="section-title">Riwayat Prediksi</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.history),
                     use_container_width=True, hide_index=True)
        if st.button("Hapus Riwayat"):
            st.session_state.history = []
            st.rerun()

# ════════════════════════════════════════════════════════
# HALAMAN 5 — BATCH PREDIKSI
# ════════════════════════════════════════════════════════
elif page == "Batch Prediksi":
    st.markdown("""
    <div class="page-header">
        <h1>Batch Prediksi</h1>
        <p>Upload file CSV berisi ulasan untuk diprediksi sentimennya sekaligus</p>
    </div>""", unsafe_allow_html=True)

    st.info("Format CSV: wajib memiliki kolom bernama **Text**")
    uploaded = st.file_uploader("Upload file CSV:", type=['csv'])

    if uploaded:
        df_up = pd.read_csv(uploaded)
        if 'Text' not in df_up.columns:
            st.error("Kolom 'Text' tidak ditemukan dalam file CSV.")
        else:
            st.success(f"{len(df_up):,} baris ditemukan.")
            st.dataframe(df_up.head(), use_container_width=True, hide_index=True)

            model_choice = st.selectbox("Model:", list(models.keys()), index=2)
            if st.button("Prediksi Semua", type="primary"):
                with st.spinner("Memproses..."):
                    df_up['clean_text'] = df_up['Text'].apply(preprocess)
                    df_up['Prediksi Sentimen'] = models[model_choice].predict(
                        vecs[model_choice].transform(df_up['clean_text'])
                    )

                st.success("Prediksi selesai.")
                col1, col2 = st.columns([1.5, 1])
                with col1:
                    st.dataframe(df_up[['Text','Prediksi Sentimen']],
                                 use_container_width=True, hide_index=True)
                with col2:
                    dist = df_up['Prediksi Sentimen'].value_counts().reset_index()
                    dist.columns = ['Sentimen', 'Jumlah']
                    fig_b = px.pie(dist, names='Sentimen', values='Jumlah',
                                   color='Sentimen', color_discrete_map=COLORS, hole=0.4)
                    fig_b.update_layout(margin=dict(t=10,b=10), height=280,
                                        legend=dict(orientation='h', y=-0.15))
                    st.plotly_chart(fig_b, use_container_width=True)

                csv_out = df_up[['Text','Prediksi Sentimen']].to_csv(index=False)
                st.download_button("Download Hasil CSV", csv_out,
                                   "hasil_prediksi.csv", "text/csv")

    st.markdown("---")
    st.caption("Belum punya file CSV? Download template berikut:")
    tpl = pd.DataFrame({'Text': [
        'This product is amazing and very fresh!',
        'It was okay, nothing special.',
        'Terrible product, complete waste of money.'
    ]})
    st.download_button("Download Template CSV", tpl.to_csv(index=False),
                       "template.csv", "text/csv")

# ════════════════════════════════════════════════════════
# HALAMAN 6 — TENTANG PROYEK
# ════════════════════════════════════════════════════════
elif page == "Tentang Proyek":
    st.markdown("""
    <div class="page-header">
        <h1>Tentang Proyek</h1>
        <p>Sentiment Classification of Amazon Fine Food Reviews Using NLP</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Informasi Proyek</div>', unsafe_allow_html=True)
        info = pd.DataFrame({
            'Keterangan': ['Mata Kuliah','Institusi','Tahun','Dataset','Algoritma','Fitur'],
            'Detail': ['Pemrosesan Bahasa Alami (NLP)',
                       'Universitas Muhammadiyah Malang',
                       '2025',
                       'Amazon Fine Food Reviews (Kaggle)',
                       'Decision Tree Classifier',
                       'Bag of Words, N-Gram, TF-IDF']
        })
        st.dataframe(info, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Tim</div>', unsafe_allow_html=True)
        tim = pd.DataFrame({
            'NIM': ['202310370311015', '202310370311014'],
            'Nama': ['Bukhary Kelian', 'Moch. Luqman Hakim']
        })
        st.dataframe(tim, use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="section-title">Referensi</div>', unsafe_allow_html=True)
        st.markdown("""
        - Amazon Fine Food Reviews — [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
        - Bird, S., et al. (2009). *Natural Language Processing with Python*. O'Reilly Media.
        - Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
        - Streamlit Documentation — [docs.streamlit.io](https://docs.streamlit.io)
        """)

        st.markdown('<div class="section-title">Ringkasan Hasil</div>', unsafe_allow_html=True)
        hasil = pd.DataFrame({
            'Model':    ['DT + BoW', 'DT + N-Gram', 'DT + TF-IDF'],
            'Accuracy': ['77.72%', '74.27%', '77.42%'],
            'F1-Score': ['0.7718', '0.7442', '0.7722'],
            'Status':   ['—', '—', 'Terbaik']
        })
        st.dataframe(hasil, use_container_width=True, hide_index=True)
