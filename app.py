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
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# =========================
# Konfigurasi Halaman & UI
# =========================
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

# =========================
# Setup & Logic (NEW NLP)
# =========================
@st.cache_resource(show_spinner="Menyiapkan NLTK...")
def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    
    # Daftar negasi yang diperluas
    negation_words = {
        "not", "no", "nor", "never", "none", "nobody", "nothing", "nowhere", "neither", "cannot",
        "ain't", "aren't", "can't", "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't", 
        "haven't", "isn't", "mightn't", "mustn't", "needn't", "shan't", "shouldn't", "wasn't", 
        "weren't", "won't", "wouldn't", "hardly", "scarcely", "barely", "rarely"
    }
    
    # Ambil stopwords NLTK dan hapus kata negasi dari daftar tersebut
    stop_words_set = set(stopwords.words('english'))
    stop_words_set = stop_words_set - negation_words
    
    return stop_words_set, WordNetLemmatizer()

stop_words, lemmatizer = setup_nltk()

def preprocess_text(text):
    text = html.unescape(str(text))
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    
    # Penanganan Singkatan (Contractions) sebelum menghapus tanda baca
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not ", text)
    
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

@st.cache_data(show_spinner="Memuat Dataset...")
def load_dataset():
    df = pd.read_csv("reviews_sample.csv")
    if 'clean_text' not in df.columns:
        df['clean_text'] = df['Text'].astype(str).apply(preprocess_text)
    return df

@st.cache_resource(show_spinner="Memuat Model...")
def load_models():
    return {
        "DT + BoW": {
            "vectorizer": joblib.load("models/vec_bow.joblib"),
            "model": joblib.load("models/dt_bow.joblib")
        },
        "DT + N-Gram": {
            "vectorizer": joblib.load("models/vec_ngram.joblib"),
            "model": joblib.load("models/dt_ngram.joblib")
        },
        "DT + TF-IDF": {
            "vectorizer": joblib.load("models/vec_tfidf.joblib"),
            "model": joblib.load("models/dt_tfidf.joblib")
        }
    }

@st.cache_data(show_spinner="Memuat Hasil Performa...")
def load_results():
    try:
        return pd.read_csv("models/model_results.csv")
    except:
        # Fallback dummy dataframe jika file model_results.csv tidak ada/error
        return pd.DataFrame({
            'Model': ['DT + BoW', 'DT + N-Gram', 'DT + TF-IDF'],
            'Accuracy': [0.7772, 0.7427, 0.7742],
            'F1-Score': [0.7718, 0.7442, 0.7722]
        })

@st.cache_data
def get_confusion_matrix(df, _models):
    # Dihitung on the fly untuk visualisasi UI agar sama seperti file lama
    model_data = _models['DT + TF-IDF']
    X = model_data['vectorizer'].transform(df['clean_text'])
    y_pred = model_data['model'].predict(X)
    return confusion_matrix(df['Sentiment'], y_pred, labels=['Positive', 'Neutral', 'Negative'])

# =========================
# Memuat Data
# =========================
df_s = load_dataset()
models = load_models()
results_df = load_results()
cm = get_confusion_matrix(df_s, models)

if 'history' not in st.session_state:
    st.session_state.history = []

best_acc = results_df['Accuracy'].max()
best_model = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]

# =========================
# Sidebar
# =========================
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
st.sidebar.markdown(f"{best_model}")
st.sidebar.markdown(f"Accuracy: {best_acc:.2%}")

# =========================
# Halaman: Overview
# =========================
if page == "Overview":
    st.markdown(f"""
    <div class="page-header">
        <h1>Sentiment Classification Dashboard</h1>
        <p>Amazon Fine Food Reviews — NLP Project | Universitas Muhammadiyah Malang 2025</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card"><div class="val">{len(df_s):,}</div><div class="lbl">Sample Data</div></div>
        <div class="metric-card"><div class="val">{best_acc:.2%}</div><div class="lbl">Best Accuracy</div></div>
        <div class="metric-card"><div class="val">3</div><div class="lbl">Model Scenarios</div></div>
        <div class="metric-card"><div class="val">{best_model}</div><div class="lbl">Best Model</div></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.markdown('<div class="section-title">Project Summary</div>', unsafe_allow_html=True)
        st.write(
            "Dashboard ini menampilkan klasifikasi sentimen ulasan produk pangan menggunakan Decision Tree "
            "dengan tiga representasi fitur yang telah melalui proses training dan disimpan secara lokal (joblib)."
        )
        st.markdown('<div class="section-title">Processing Pipeline</div>', unsafe_allow_html=True)
        pipeline = pd.DataFrame({
            'Stage': ['Load', 'Clean', 'Preprocess', 'Vectorize', 'Predict', 'Evaluate'],
            'Main Step': ['Load Joblib Models', 'URL & Regex Clean', 'Lowercase, stopwords, lemmatize',
                          'BoW / N-Gram / TF-IDF', 'Decision Tree', 'Accuracy, F1, Confusion Matrix']
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

# =========================
# Halaman: Prediksi Sentimen
# =========================
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
            clean = preprocess_text(user_input)
            vec = models[model_choice]["vectorizer"]
            model = models[model_choice]["model"]
            
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

# =========================
# Halaman: Batch Prediksi
# =========================
if page == "Batch Prediksi":
    st.markdown(f"""
    <div class="page-header">
        <h1>Batch Prediksi & Evaluasi</h1>
        <p>Upload CSV berisi ulasan untuk diprediksi secara massal atau dievaluasi performanya</p>
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
            
            st.success(f"{len(df_up):,} baris (instances) berhasil dibaca.")
            
            # FITUR EVALUASI
            eval_mode = st.checkbox("File ini memiliki label asli (Aktifkan untuk Evaluasi Model & Analisis Error)")
            
            col1, col2 = st.columns(2)
            with col1:
                text_col = st.selectbox("Pilih kolom Teks Ulasan:", df_up.columns)
            
            with col2:
                if eval_mode:
                    label_col = st.selectbox("Pilih kolom Label Asli (Ground Truth):", df_up.columns)
                else:
                    st.info("Mode prediksi buta (tanpa label asli).")

            df_up["Text"] = df_up[text_col].astype(str).fillna("").str.strip()
            df_up = df_up[df_up["Text"] != ""]

            if df_up.empty:
                st.warning("Tidak ada teks valid untuk diprediksi.")
            else:
                model_choice = st.selectbox("Pilih model", list(models.keys()), index=2)

                if st.button("Jalankan Proses", use_container_width=True):
                    with st.spinner("Memproses teks dan memprediksi..."):
                        # Preprocessing text
                        df_up["clean_text"] = df_up["Text"].apply(preprocess_text)
                        
                        vec = models[model_choice]["vectorizer"]
                        model_obj = models[model_choice]["model"]
                        
                        # Prediksi
                        X_batch = vec.transform(df_up["clean_text"])
                        df_up["Prediksi Sentimen"] = model_obj.predict(X_batch)
                        
                        # FITUR BARU: Ambil nilai Confidence (Probabilitas)
                        proba = model_obj.predict_proba(X_batch)
                        df_up["Confidence Score"] = proba.max(axis=1)
                        
                    st.success("Proses selesai!")

                    # ==========================================
                    # BAGIAN EVALUASI (CONFUSION MATRIX & REPORT)
                    # ==========================================
                    if eval_mode:
                        st.markdown('<div class="section-title">Hasil Evaluasi Model</div>', unsafe_allow_html=True)
                        
                        y_true = df_up[label_col].astype(str).str.capitalize()
                        y_pred = df_up["Prediksi Sentimen"]
                        
                        # 1. Total Instances & Accuracy
                        acc = accuracy_score(y_true, y_pred)
                        c_met1, c_met2 = st.columns(2)
                        c_met1.metric("Total Instances", f"{len(y_true):,}")
                        c_met2.metric("Akurasi Testing", f"{acc:.2%}")
                        
                        c_rep, c_cm = st.columns(2)
                        
                        # 2. Classification Report
                        with c_rep:
                            st.markdown("**Classification Report**")
                            report_dict = classification_report(y_true, y_pred, output_dict=True)
                            report_df = pd.DataFrame(report_dict).transpose()
                            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
                            
                        # 3. Confusion Matrix
                        with c_cm:
                            st.markdown("**Confusion Matrix**")
                            labels_cm = ['Positive', 'Neutral', 'Negative']
                            cm_batch = confusion_matrix(y_true, y_pred, labels=labels_cm)
                            
                            fig_cm = px.imshow(cm_batch, text_auto=True, x=labels_cm, y=labels_cm,
                                               color_continuous_scale=[[0, '#fff5f5'], [1, MAROON]])
                            fig_cm.update_layout(height=320, margin=dict(t=10, b=10))
                            st.plotly_chart(fig_cm, use_container_width=True)
                            
                        # 4. FITUR BARU: Tabel Analisis Kesalahan (Error Analysis)
                        st.markdown('<div class="section-title">Analisis Kesalahan (False Positives / False Negatives)</div>', unsafe_allow_html=True)
                        error_df = df_up[y_true != y_pred].copy()
                        
                        if not error_df.empty:
                            st.warning(f"Terdapat {len(error_df)} data yang salah diprediksi oleh model. Berikut detailnya:")
                            # Tampilkan kolom teks, label asli, tebakan, dan confidence
                            st.dataframe(error_df[[text_col, label_col, "Prediksi Sentimen", "Confidence Score"]], 
                                         use_container_width=True, hide_index=True)
                        else:
                            st.success("Luar Biasa! Model berhasil menebak semua data dengan akurat tanpa kesalahan.")
                        st.markdown("---")

                    # ==========================================
                    # BAGIAN GRAFIK & DATA TABEL (STANDAR)
                    # ==========================================
                    col_pie, col_hist = st.columns(2)
                    
                    with col_pie:
                        st.markdown('<div class="section-title">Distribusi Sentimen</div>', unsafe_allow_html=True)
                        dist = df_up["Prediksi Sentimen"].value_counts().reset_index()
                        dist.columns = ["Sentiment", "Count"]
                        fig_pie = px.pie(dist, names="Sentiment", values="Count", hole=0.4,
                                     color="Sentiment", color_discrete_map=COLORS)
                        fig_pie.update_layout(height=300, margin=dict(t=10, b=10))
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col_hist:
                        # FITUR BARU: Grafik Distribusi Confidence Score
                        st.markdown('<div class="section-title">Keyakinan Model (Confidence)</div>', unsafe_allow_html=True)
                        fig_hist = px.histogram(df_up, x="Confidence Score", color="Prediksi Sentimen", nbins=20,
                                                color_discrete_map=COLORS, barmode="overlay", opacity=0.75)
                        fig_hist.update_layout(height=300, margin=dict(t=10, b=10), xaxis_title="Confidence Score (0 - 1.0)")
                        st.plotly_chart(fig_hist, use_container_width=True)

                    # Tabel Seluruh Hasil
                    st.markdown('<div class="section-title">Tabel Lengkap Hasil Prediksi</div>', unsafe_allow_html=True)
                    
                    # Format nilai Confidence agar tampil sebagai persentase di tabel
                    df_tampil = df_up.copy()
                    df_tampil["Confidence Score"] = df_tampil["Confidence Score"].apply(lambda x: f"{x:.2%}")
                    
                    tampil_kolom = ["Text", "Prediksi Sentimen", "Confidence Score"]
                    if eval_mode:
                        tampil_kolom.insert(1, label_col) # Tampilkan label asli jika ada
                        
                    st.dataframe(df_tampil[tampil_kolom], use_container_width=True, hide_index=True)

                    # Export
                    csv_out = df_tampil[tampil_kolom].to_csv(index=False)
                    st.download_button("Download Hasil Lengkap", csv_out, "hasil_batch_prediksi_lengkap.csv", "text/csv")

# =========================
# Halaman: Performa Model
# =========================
if page == "Performa Model":
    st.markdown(f"""
    <div class="page-header">
        <h1>Performa Model</h1>
        <p>Perbandingan hasil metriks model dan Confusion Matrix (berdasarkan model terbaik)</p>
    </div>
    """, unsafe_allow_html=True)

    # Menampilkan DataFrame hasil berdasarkan model_results.csv
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # Plot metrics
    y_cols = [col for col in ['Accuracy', 'F1-Score'] if col in results_df.columns]
    if y_cols:
        fig = px.bar(results_df, x='Model', y=y_cols, barmode='group',
                     color_discrete_sequence=[MAROON, GOLD])
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Confusion Matrix (DT + TF-IDF)</div>', unsafe_allow_html=True)
    labels = ['Positive', 'Neutral', 'Negative']
    fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels,
                       color_continuous_scale=[[0, '#fff5f5'], [1, MAROON]])
    fig_cm.update_layout(height=320, margin=dict(t=10, b=10))
    st.plotly_chart(fig_cm, use_container_width=True)

# =========================
# Halaman: Eksplorasi Data
# =========================
if page == "Eksplorasi Data":
    st.markdown(f"""
    <div class="page-header">
        <h1>Eksplorasi Data</h1>
        <p>Distribusi label, rating, panjang ulasan, dan word cloud</p>
    </div>
    """, unsafe_allow_html=True)

    if 'Sentiment' in df_s.columns:
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
            if 'Score' in dff.columns:
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
                if len(subset) > 0:
                    txt = ' '.join(subset.sample(min(800, len(subset)), random_state=42))
                    wc = WordCloud(width=450, height=280, background_color='white', colormap=cmap).generate(txt)
                    fig_wc, ax = plt.subplots(figsize=(5, 3.2))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig_wc)

        n = st.slider("Jumlah baris tabel", 10, 50, 10)
        display_cols = [c for c in ['Score', 'Sentiment', 'Summary', 'Text'] if c in dff.columns]
        st.dataframe(dff[display_cols].head(n),
                     use_container_width=True, hide_index=True)

# =========================
# Halaman: Tentang Proyek
# =========================
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
                       'BoW, N-Gram, TF-IDF', best_model]
        })
        st.dataframe(info, use_container_width=True, hide_index=True)
    with c2:
        st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(f'<div class="section-title">Anggota Tim</div>', unsafe_allow_html=True)

    tim_df = pd.DataFrame({
        'Nama': ['Bukhary Kelian', 'Moch. Luqman Hakim'],
        'NIM': ['202310370311015', '202310370311014']
    })
    st.dataframe(tim_df, use_container_width=True, hide_index=True)
