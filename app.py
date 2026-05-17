import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import re
import html
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
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
COLORS = {
    'Positive': '#27ae60',
    'Neutral': '#e67e22',
    'Negative': '#c0392b'
}

st.markdown(f"""
<style>
.main {{
    background-color: #fdfaf7;
}}
.block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
}}
h1, h2, h3 {{
    color: {MAROON};
    font-weight: 700;
}}
.sidebar .sidebar-content {{
    background-color: #faf6f2;
}}
div[data-testid="metric-container"] {{
    background: white;
    border: 1px solid #eadfda;
    padding: 16px;
    border-radius: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}}
.stButton > button {{
    background-color: {MAROON};
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}}
.stButton > button:hover {{
    background-color: #5f0000;
    color: white;
}}
.stDownloadButton > button {{
    background-color: {GOLD};
    color: black;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}}
section[data-testid="stSidebar"] {{
    background-color: #f8f1eb;
}}
hr {{
    border: none;
    border-top: 1px solid #e5d7d0;
    margin: 1rem 0;
}}
.card {{
    background: white;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #eadfda;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}}
.small-note {{
    color: {GRAY};
    font-size: 0.9rem;
}}
</style>
""", unsafe_allow_html=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = html.unescape(str(text)).lower()
    text = re.sub(r"http\\S+|www\\S+|https\\S+", " ", text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

@st.cache_data
def load_dataset():
    df = pd.read_csv("reviews_sample.csv")
    if 'clean_text' not in df.columns:
        df['clean_text'] = df['Text'].astype(str).apply(preprocess)
    return df

@st.cache_resource(show_spinner="Memuat model...")
def load_artifacts():
    vecs = {
        'DT + BoW': joblib.load("models/vec_bow.joblib"),
        'DT + N-Gram': joblib.load("models/vec_ngram.joblib"),
        'DT + TF-IDF': joblib.load("models/vec_tfidf.joblib")
    }

    models = {
        'DT + BoW': joblib.load("models/dt_bow.joblib"),
        'DT + N-Gram': joblib.load("models/dt_ngram.joblib"),
        'DT + TF-IDF': joblib.load("models/dt_tfidf.joblib")
    }

    results = pd.read_csv("models/model_results.csv")
    return vecs, models, results

df_s = load_dataset()
vecs, models, results_df = load_artifacts()

if 'history' not in st.session_state:
    st.session_state.history = []

best_idx = results_df["Accuracy"].idxmax()
best_model = results_df.loc[best_idx, "Model"]
best_acc = results_df.loc[best_idx, "Accuracy"]

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
st.sidebar.markdown(best_model)
st.sidebar.markdown(f"Accuracy: {best_acc:.2%}")

if page == "Overview":
    st.markdown(f"# NLP Sentiment Dashboard")
    st.markdown("Amazon Fine Food Reviews — NLP Project | Universitas Muhammadiyah Malang 2025")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Data", f"{len(df_s):,}")
    c2.metric("Jumlah Model", "3")
    c3.metric("Model Terbaik", best_model)
    c4.metric("Accuracy Terbaik", f"{best_acc:.4f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Sentimen")
        sentiment_counts = df_s["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            color="Sentiment",
            color_discrete_map=COLORS,
            template="simple_white"
        )
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#333"),
            title=None
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Perbandingan Accuracy Model")
        fig2 = px.bar(
            results_df,
            x="Model",
            y="Accuracy",
            color="Model",
            color_discrete_sequence=[MAROON, GOLD, "#b85c5c"],
            template="simple_white"
        )
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Prediksi Sentimen":
    st.markdown("# Prediksi Sentimen")

    model_choice = st.selectbox("Pilih Model", list(models.keys()))
    text_input = st.text_area("Input satu ulasan", height=180)

    if st.button("Prediksi"):
        if text_input.strip():
            cleaned = preprocess(text_input)
            X_input = vecs[model_choice].transform([cleaned])

            pred = models[model_choice].predict(X_input)[0]

            proba = None
            if hasattr(models[model_choice], "predict_proba"):
                proba = models[model_choice].predict_proba(X_input)[0]

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write(f"**Sentimen:** {pred}")
            st.write(f"**Model:** {model_choice}")
            if proba is not None:
                st.write(f"**Confidence:** {max(proba):.1%}")
            st.write(f"**Preprocessed Text:** {cleaned}")
            st.markdown('</div>', unsafe_allow_html=True)

            st.session_state.history.append({
                "Text": text_input,
                "Prediction": pred,
                "Model": model_choice
            })
        else:
            st.warning("Masukkan teks terlebih dahulu.")

elif page == "Batch Prediksi":
    st.markdown("# Batch Prediksi")

    model_choice = st.selectbox("Pilih Model untuk Batch", list(models.keys()))
    uploaded_file = st.file_uploader("Upload CSV berisi kolom teks", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.dataframe(batch_df.head(), use_container_width=True)

        text_col = st.selectbox("Pilih kolom teks", batch_df.columns)

        if st.button("Proses Batch"):
            batch_df["clean_text"] = batch_df[text_col].astype(str).apply(preprocess)
            X_batch = vecs[model_choice].transform(batch_df["clean_text"])
            batch_df["Predicted_Sentiment"] = models[model_choice].predict(X_batch)

            st.success("Batch prediction selesai.")
            st.dataframe(batch_df.head(20), use_container_width=True)

            csv = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Hasil CSV",
                data=csv,
                file_name="hasil_prediksi_sentimen.csv",
                mime="text/csv"
            )

elif page == "Performa Model":
    st.markdown("# Performa Model")

    st.subheader("Ringkasan Accuracy")
    st.dataframe(results_df, use_container_width=True)

    selected_eval_model = st.selectbox("Pilih model evaluasi", list(models.keys()))

    X_eval = vecs[selected_eval_model].transform(df_s["clean_text"])
    y_true = df_s["Sentiment"]
    y_pred = models[selected_eval_model].predict(X_eval)

    cm = confusion_matrix(y_true, y_pred, labels=['Positive', 'Neutral', 'Negative'])

    fig_cm, ax = plt.subplots(figsize=(6, 5))
    disp = plt.imshow(cm, cmap="OrRd")
    plt.xticks(range(3), ['Positive', 'Neutral', 'Negative'])
    plt.yticks(range(3), ['Positive', 'Neutral', 'Negative'])
    plt.colorbar(disp)
    plt.title(f'Confusion Matrix - {selected_eval_model}')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig_cm)

elif page == "Eksplorasi Data":
    st.markdown("# Eksplorasi Data")

    st.subheader("Preview Dataset")
    st.dataframe(df_s.head(20), use_container_width=True)

    st.subheader("Distribusi Rating")
    if "Score" in df_s.columns:
        score_counts = df_s["Score"].value_counts().sort_index().reset_index()
        score_counts.columns = ["Score", "Count"]
        fig_score = px.bar(score_counts, x="Score", y="Count", color_discrete_sequence=[GOLD])
        st.plotly_chart(fig_score, use_container_width=True)

    st.subheader("Word Cloud")
    all_text = " ".join(df_s["clean_text"].dropna().astype(str).tolist()[:3000])
    wc = WordCloud(width=1000, height=400, background_color="white", colormap="OrRd").generate(all_text)

    fig_wc, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig_wc)

elif page == "Tentang Proyek":
    st.markdown("# Tentang Proyek")
    st.markdown("""
    <div class="card">
    <h4 style="color:#800000;">Mini Project NLP</h4>
    <p>
    Dashboard ini digunakan untuk analisis sentimen pada dataset Amazon Fine Food Reviews
    dengan tiga pendekatan representasi teks:
    Bag of Words, N-Gram, dan TF-IDF, menggunakan classifier Decision Tree.
    </p>
    <p>
    Versi ini sudah menggunakan pendekatan deployment-ready, yaitu model dan vectorizer
    dimuat dari file artifact `.joblib`, bukan retraining saat runtime.
    </p>
    </div>
    """, unsafe_allow_html=True)
