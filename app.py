import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import re
import html
import nltk
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

st.set_page_config(
    page_title="NLP Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

MAROON = "#800000"
GOLD = "#c9a84c"
GRAY = "#6b6b6b"
COLORS = {'Positive': '#27ae60', 'Neutral': '#e67e22', 'Negative': '#c0392b'}

st.markdown(f"""
<style>
html, body, [class*="css"] {{ font-family: 'Segoe UI', sans-serif; }}
section[data-testid="stSidebar"] {{ background-color: {MAROON}; }}
section[data-testid="stSidebar"] * {{ color: white !important; }}

.page-header {{
    background: linear-gradient(135deg, {MAROON} 0%, #4a0000 100%);
    padding: 22px 26px;
    border-radius: 12px;
    color: white;
    margin-bottom: 18px;
}}
.page-header h1 {{
    margin: 0;
    font-size: 1.7rem;
    font-weight: 700;
}}
.page-header p {{
    margin: 6px 0 0 0;
    font-size: 0.92rem;
    opacity: 0.88;
}}

.metric-row {{
    display: flex;
    gap: 12px;
    margin-bottom: 18px;
}}
.metric-card {{
    flex: 1;
    background: white;
    border-left: 5px solid {MAROON};
    border-radius: 10px;
    padding: 16px 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}}
.metric-card .val {{
    font-size: 1.7rem;
    font-weight: 700;
    color: {MAROON};
    line-height: 1.1;
}}
.metric-card .lbl {{
    font-size: 0.8rem;
    color: {GRAY};
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}}

.section-title {{
    font-size: 1rem;
    font-weight: 700;
    color: {MAROON};
    border-bottom: 2px solid {MAROON};
    padding-bottom: 5px;
    margin: 18px 0 12px 0;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}}

.result-positive {{
    background: #eaf7ef;
    border-left: 6px solid #27ae60;
    border-radius: 10px;
    padding: 16px 18px;
    margin: 14px 0;
}}
.result-negative {{
    background: #fdecea;
    border-left: 6px solid #c0392b;
    border-radius: 10px;
    padding: 16px 18px;
    margin: 14px 0;
}}
.result-neutral {{
    background: #fef9ec;
    border-left: 6px solid #e67e22;
    border-radius: 10px;
    padding: 16px 18px;
    margin: 14px 0;
}}
.result-label {{
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0;
}}
.result-sub {{
    font-size: 0.88rem;
    color: {GRAY};
    margin-top: 4px;
}}
</style>
""", unsafe_allow_html=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = html.unescape(str(text)).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

@st.cache_resource(show_spinner="Memuat model...")
def load_and_train():
    df = pd.read_csv('reviews_sample.csv')
    if 'clean_text' not in df.columns:
        df['clean_text'] = df['Text'].apply(preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['Sentiment'],
        test_size=0.2, stratify=df['Sentiment'], random_state=42
    )

    vecs = {
        'DT + BoW': CountVectorizer(max_features=10000),
        'DT + N-Gram': CountVectorizer(max_features=10000, ngram_range=(2, 2)),
        'DT + TF-IDF': TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    }

    models, reports = {}, {}
    for name, vec in vecs.items():
        model = DecisionTreeClassifier(random_state=42)
        Xtr = vec.fit_transform(X_train)
        Xte = vec.transform(X_test)
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        models[name] = model
        reports[name] = classification_report(y_test, pred, output_dict=True)

    tfidf_pred = models['DT + TF-IDF'].predict(vecs['DT + TF-IDF'].transform(X_test))
    cm = confusion_matrix(y_test, tfidf_pred, labels=['Positive', 'Neutral', 'Negative'])
    return df, vecs, models, reports, cm

df_s, vecs, models, reports, cm = load_and_train()

if 'history' not in st.session_state:
    st.session_state.history = []

st.sidebar.markdown("## NLP Sentiment Dashboard")
st.sidebar.markdown("Amazon Fine Food Reviews")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigasi", [
    "Overview",
    "Prediksi Sentimen",
    "Batch Prediksi",
    "Performa Model",
    "Eksplorasi Data",
    "Tentang Proyek"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Terbaik**")
st.sidebar.markdown("DT + TF-IDF")
st.sidebar.markdown("Accuracy: 77.42%")
st.sidebar.markdown("F1: 0.7722")

if page == "Overview":
    st.markdown(f"""
    <div class="page-header">
        <h1>Sentiment Classification Dashboard</h1>
        <p>Amazon Fine Food Reviews — NLP Project | Universitas Muhammadiyah Malang 2025</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card"><div class="val">50.000</div><div class="lbl">Sample Data</div></div>
        <div class="metric-card"><div class="val">77.42%</div><div class="lbl">Best Accuracy</div></div>
        <div class="metric-card"><div class="val">0.7722</div><div class="lbl">Best F1-Score</div></div>
        <div class="metric-card"><div class="val">3</div><div class="lbl">Model Experiments</div></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.markdown('<div class="section-title">Project Summary</div>', unsafe_allow_html=True)
        st.write(
            "Dashboard ini menampilkan klasifikasi sentimen ulasan produk pangan menggunakan Decision Tree "
            "dengan tiga representasi fitur: Bag of Words, N-Gram, dan TF-IDF."
        )
        st.markdown('<div class="section-title">Processing Pipeline</div>', unsafe_allow_html=True)
        pipeline = pd.DataFrame({
            'Stage': ['Load', 'Clean', 'Preprocess', 'Vectorize', 'Train', 'Evaluate'],
            'Main Step': ['Read sample CSV', 'Missing + duplicate removal', 'Lowercase, stopwords, lemmatize',
                          'BoW / N-Gram / TF-IDF', 'Decision Tree', 'Accuracy, Precision, Recall, F1']
        })
        st.dataframe(pipeline, use_container_width=True, hide_index=True)

    with c2:
        st.markdown('<div class="section-title">Label Distribution</div>', unsafe_allow_html=True)
        counts = df_s['Sentiment'].value_counts().reset_index()
        counts.columns = ['Sentiment', 'Count']
        fig = px.pie(
            counts, names='Sentiment', values='Count', hole=0.4,
            color='Sentiment', color_discrete_map=COLORS
        )
        fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10),
                          legend=dict(orientation='h', y=-0.12))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">First 10 Records</div>', unsafe_allow_html=True)
    st.dataframe(df_s[['Score', 'Sentiment', 'Summary', 'Text']].head(10),
                 use_container_width=True, hide_index=True)

if page == "Prediksi Sentimen":
    st.markdown(f"""
    <div class="page-header">
        <h1>Prediksi Sentimen</h1>
        <p>Input satu ulasan untuk prediksi sentimen secara real-time</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.4, 1])
    with col1:
        model_choice = st.selectbox("Pilih model", list(models.keys()), index=2)
        user_input = st.text_area("Teks ulasan", height=140, placeholder="Contoh: This product is amazing...")
        submit = st.button("Prediksi", use_container_width=True)
    with col2:
        st.markdown('<div class="section-title">Contoh Input</div>', unsafe_allow_html=True)
        st.write("Positif: This product is absolutely amazing. Great taste and very fresh.")
        st.write("Netral: It was okay. Nothing special but not bad either.")
        st.write("Negatif: Terrible product. Arrived damaged and tasted awful.")

    if submit:
        if not user_input.strip():
            st.warning("Masukkan teks ulasan terlebih dahulu.")
        else:
            clean = preprocess(user_input)
            vec = vecs[model_choice]
            model = models[model_choice]
            x = vec.transform([clean])
            pred = model.predict(x)[0]
            proba = model.predict_proba(x)[0]
            classes = model.classes_

            cls = {'Positive': 'result-positive', 'Neutral': 'result-neutral', 'Negative': 'result-negative'}[pred]
            st.markdown(f"""
            <div class="{cls}">
                <p class="result-label">Sentimen: {pred}</p>
                <p class="result-sub">Model: {model_choice} | Confidence: {max(proba):.1%}</p>
            </div>
            """, unsafe_allow_html=True)

            prob_df = pd.DataFrame({'Class': classes, 'Confidence': proba})
            fig = px.bar(prob_df, x='Class', y='Confidence', color='Class',
                         color_discrete_map=COLORS, range_y=[0, 1])
            fig.update_layout(height=300, margin=dict(t=10, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Hasil preprocessing"):
                st.code(clean)

            st.session_state.history.insert(0, {
                'Model': model_choice,
                'Input': user_input[:80] + ('...' if len(user_input) > 80 else ''),
                'Prediction': pred,
                'Confidence': f"{max(proba):.1%}"
            })

    if st.session_state.history:
        st.markdown('<div class="section-title">Riwayat Prediksi</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.history).head(5),
                     use_container_width=True, hide_index=True)

if page == "Batch Prediksi":
    st.markdown(f"""
    <div class="page-header">
        <h1>Batch Prediksi</h1>
        <p>Upload CSV berisi kolom Text untuk prediksi banyak ulasan sekaligus</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload file CSV", type=["csv"])

    def read_uploaded_csv(file):
        for kwargs in [
            dict(sep=None, engine="python"),
            dict(sep=",", engine="python"),
            dict(sep=";", engine="python"),
        ]:
            try:
                file.seek(0)
                df = pd.read_csv(file, **kwargs)
                if df.shape[1] == 1 and "Text" not in df.columns:
                    continue
                return df
            except Exception:
                continue
        return None

    if uploaded is not None:
        df_up = read_uploaded_csv(uploaded)

        if df_up is None:
            st.error("File tidak bisa dibaca sebagai CSV. Cek format delimiter dan header.")
        else:
            df_up.columns = [str(c).strip() for c in df_up.columns]

            if "Text" not in df_up.columns:
                if df_up.shape[1] == 1:
                    df_up = df_up.rename(columns={df_up.columns[0]: "Text"})
                else:
                    st.error("Kolom 'Text' tidak ditemukan. Pastikan header kolom bernama Text.")
                    st.stop()

            df_up["Text"] = df_up["Text"].astype(str).fillna("").str.strip()
            df_up = df_up[df_up["Text"] != ""]

            if df_up.empty:
                st.warning("Tidak ada teks valid untuk diprediksi.")
            else:
                st.success(f"{len(df_up):,} baris berhasil dibaca.")
                st.dataframe(df_up.head(), use_container_width=True, hide_index=True)

                model_choice = st.selectbox("Pilih model batch", list(models.keys()), index=2)

                if st.button("Prediksi Semua", use_container_width=True):
                    with st.spinner("Memproses..."):
                        df_up["clean_text"] = df_up["Text"].apply(preprocess)
                        X_batch = vecs[model_choice].transform(df_up["clean_text"])
                        df_up["Prediksi Sentimen"] = models[model_choice].predict(X_batch)

                    st.success("Prediksi selesai.")
                    st.dataframe(df_up[["Text", "Prediksi Sentimen"]],
                                 use_container_width=True, hide_index=True)

                    dist = df_up["Prediksi Sentimen"].value_counts().reset_index()
                    dist.columns = ["Sentiment", "Count"]
                    fig = px.pie(dist, names="Sentiment", values="Count", hole=0.4,
                                 color="Sentiment", color_discrete_map=COLORS)
                    fig.update_layout(height=300, margin=dict(t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                    csv_out = df_up[["Text", "Prediksi Sentimen"]].to_csv(index=False)
                    st.download_button("Download Hasil", csv_out, "hasil_prediksi.csv", "text/csv")

if page == "Performa Model":
    st.markdown(f"""
    <div class="page-header">
        <h1>Performa Model</h1>
        <p>Perbandingan DT + BoW, DT + N-Gram, dan DT + TF-IDF</p>
    </div>
    """, unsafe_allow_html=True)

    perf = []
    for name, r in reports.items():
        perf.append({
            'Model': name,
            'Accuracy': round(r['accuracy'], 4),
            'Precision': round(r['weighted avg']['precision'], 4),
            'Recall': round(r['weighted avg']['recall'], 4),
            'F1-Score': round(r['weighted avg']['f1-score'], 4)
        })
    perf_df = pd.DataFrame(perf)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    fig = px.bar(perf_df, x='Model', y=['Accuracy', 'F1-Score'], barmode='group',
                 color_discrete_sequence=[MAROON, GOLD])
    fig.update_layout(height=320, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    labels = ['Positive', 'Neutral', 'Negative']
    fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels,
                       color_continuous_scale=[[0, '#fff5f5'], [1, MAROON]])
    fig_cm.update_layout(height=320, margin=dict(t=10, b=10))
    st.plotly_chart(fig_cm, use_container_width=True)

if page == "Eksplorasi Data":
    st.markdown(f"""
    <div class="page-header">
        <h1>Eksplorasi Data</h1>
        <p>Distribusi label, rating, panjang ulasan, dan word cloud</p>
    </div>
    """, unsafe_allow_html=True)

    filt = st.multiselect("Filter Sentimen", ['Positive', 'Neutral', 'Negative'],
                          default=['Positive', 'Neutral', 'Negative'])
    dff = df_s[df_s['Sentiment'].isin(filt)]

    c1, c2 = st.columns(2)
    with c1:
        a = dff['Sentiment'].value_counts().reset_index()
        a.columns = ['Sentiment', 'Count']
        fig = px.bar(a, x='Sentiment', y='Count', color='Sentiment',
                     color_discrete_map=COLORS, text='Count')
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(height=320, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        b = dff['Score'].value_counts().sort_index().reset_index()
        b.columns = ['Score', 'Count']
        fig = px.bar(b, x='Score', y='Count', color_discrete_sequence=[MAROON], text='Count')
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(height=320, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    dff2 = dff.copy()
    dff2['Length'] = dff2['Text'].astype(str).str.split().str.len()
    fig = px.histogram(dff2, x='Length', color='Sentiment', nbins=60,
                       color_discrete_map=COLORS, opacity=0.75)
    fig.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    wc1, wc2, wc3 = st.columns(3)
    for col, sent, cmap in zip([wc1, wc2, wc3], ['Positive', 'Neutral', 'Negative'], ['Greens', 'Oranges', 'Reds']):
        with col:
            subset = df_s[df_s['Sentiment'] == sent]['clean_text']
            txt = ' '.join(subset.sample(min(800, len(subset)), random_state=42))
            wc = WordCloud(width=450, height=280, background_color='white', colormap=cmap).generate(txt)
            fig_wc, ax = plt.subplots(figsize=(5, 3.2))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig_wc)

    n = st.slider("Jumlah baris tabel", 10, 50, 10)
    st.dataframe(dff[['Score', 'Sentiment', 'Summary', 'Text']].head(n),
                 use_container_width=True, hide_index=True)

if page == "Tentang Proyek":
    st.markdown(f"""
    <div class="page-header">
        <h1>Tentang Proyek</h1>
        <p>Ringkasan sistem klasifikasi sentimen</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        info = pd.DataFrame({
            'Item': ['Mata Kuliah', 'Institusi', 'Dataset', 'Algoritma', 'Fitur', 'Model Terbaik'],
            'Detail': ['Pemrosesan Bahasa Alami (NLP)', 'Universitas Muhammadiyah Malang',
                       'Amazon Fine Food Reviews', 'Decision Tree Classifier',
                       'BoW, N-Gram, TF-IDF', 'DT + TF-IDF']
        })
        st.dataframe(info, use_container_width=True, hide_index=True)
    with c2:
        hasil = pd.DataFrame({
            'Model': ['DT + BoW', 'DT + N-Gram', 'DT + TF-IDF'],
            'Accuracy': ['77.72%', '74.27%', '77.42%'],
            'F1-Score': ['0.7718', '0.7442', '0.7722']
        })
        st.dataframe(hasil, use_container_width=True, hide_index=True)

    # ========== TAMBAHAN NAMA & NIM ==========
    st.markdown("---")  # garis pemisah
    st.markdown("### Anggota Tim")

    tim_df = pd.DataFrame({
        'Nama': ['Bukhary Kelian', 'Moch. Luqman Hakim'],
        'NIM': ['202310370311015', '202310370311014']
    })
    st.dataframe(tim_df, use_container_width=True, hide_index=True)
