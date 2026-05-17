import streamlit as st
import pandas as pd
import joblib
import re
import html
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(
    page_title="NLP Sentiment Dashboard",
    page_icon="🧠",
    layout="wide"
)

# =========================
# NLTK setup
# =========================
@st.cache_resource
def setup_nltk():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    return set(stopwords.words('english')), WordNetLemmatizer()

stop_words, lemmatizer = setup_nltk()

# =========================
# Preprocessing
# =========================
def preprocess_text(text):
    text = html.unescape(str(text))
    text = text.lower()
    text = re.sub(r"http\\S+|www\\S+|https\\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return " ".join(tokens)

# =========================
# Load data and artifacts
# =========================
@st.cache_data
def load_dataset():
    df = pd.read_csv("reviews_sample.csv")
    return df

@st.cache_resource
def load_models():
    models = {
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
    return models

@st.cache_data
def load_results():
    return pd.read_csv("models/model_results.csv")

df = load_dataset()
models = load_models()
results_df = load_results()

# =========================
# Helper functions
# =========================
def predict_sentiment(text, selected_model):
    clean_text = preprocess_text(text)
    vectorizer = models[selected_model]["vectorizer"]
    model = models[selected_model]["model"]

    vectorized_text = vectorizer.transform([clean_text])
    prediction = model.predict(vectorized_text)[0]

    return prediction, clean_text

def get_label_color(label):
    if label == "Positive":
        return "green"
    elif label == "Neutral":
        return "orange"
    return "red"

# =========================
# Sidebar
# =========================
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio(
    "Pilih Halaman",
    ["Dashboard", "Single Prediction", "Batch Prediction", "Model Performance", "Dataset Explorer", "About"]
)

selected_model = st.sidebar.selectbox(
    "Pilih Model",
    ["DT + BoW", "DT + N-Gram", "DT + TF-IDF"]
)

# =========================
# Dashboard
# =========================
if menu == "Dashboard":
    st.title("🧠 NLP Sentiment Analysis Dashboard")
    st.markdown("Dashboard analisis sentimen review menggunakan beberapa skenario vectorizer dan model Decision Tree.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data", len(df))
    col2.metric("Jumlah Model", 3)
    col3.metric("Default Model", "DT + BoW")
    col4.metric("Best Accuracy", f"{results_df['Accuracy'].max():.4f}")

    st.subheader("Distribusi Sentimen Dataset")
    if "Sentiment" in df.columns:
        sentiment_counts = df["Sentiment"].value_counts()
        st.bar_chart(sentiment_counts)

    st.subheader("Ringkasan Performa Model")
    st.dataframe(results_df, use_container_width=True)

# =========================
# Single Prediction
# =========================
elif menu == "Single Prediction":
    st.title("🔍 Single Text Prediction")

    user_input = st.text_area(
        "Masukkan teks review:",
        height=200,
        placeholder="Contoh: This product is amazing and works really well..."
    )

    if st.button("Prediksi Sentimen"):
        if user_input.strip() == "":
            st.warning("Masukkan teks terlebih dahulu.")
        else:
            prediction, clean_text = predict_sentiment(user_input, selected_model)
            color = get_label_color(prediction)

            st.subheader("Hasil Prediksi")
            st.markdown(f"**Model:** {selected_model}")
            st.markdown(f"**Preprocessed Text:** {clean_text}")
            st.markdown(f"**Predicted Sentiment:** :{color}[{prediction}]")

# =========================
# Batch Prediction
# =========================
elif menu == "Batch Prediction":
    st.title("📂 Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload file CSV yang memiliki kolom teks",
        type=["csv"]
    )

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview data:")
        st.dataframe(batch_df.head(), use_container_width=True)

        text_column = st.selectbox("Pilih kolom teks", batch_df.columns)

        if st.button("Proses Batch Prediction"):
            batch_df["clean_text"] = batch_df[text_column].astype(str).apply(preprocess_text)

            vectorizer = models[selected_model]["vectorizer"]
            model = models[selected_model]["model"]

            X_batch = vectorizer.transform(batch_df["clean_text"])
            batch_df["Predicted_Sentiment"] = model.predict(X_batch)

            st.success("Batch prediction selesai.")
            st.dataframe(batch_df.head(20), use_container_width=True)

            csv_result = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Hasil Prediksi",
                data=csv_result,
                file_name="hasil_prediksi_sentimen.csv",
                mime="text/csv"
            )

# =========================
# Model Performance
# =========================
elif menu == "Model Performance":
    st.title("📊 Model Performance")

    st.subheader("Perbandingan Accuracy")
    st.dataframe(results_df, use_container_width=True)
    st.bar_chart(results_df.set_index("Model"))

    best_model = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
    best_acc = results_df["Accuracy"].max()

    st.info(f"Model terbaik berdasarkan accuracy adalah **{best_model}** dengan nilai **{best_acc:.4f}**.")

# =========================
# Dataset Explorer
# =========================
elif menu == "Dataset Explorer":
    st.title("🗂 Dataset Explorer")

    st.subheader("Preview Dataset")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Informasi Dataset")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum().values,
        "Data Type": df.dtypes.values.astype(str)
    })
    st.dataframe(info_df, use_container_width=True)

    if "Sentiment" in df.columns:
        st.subheader("Distribusi Label Sentimen")
        st.write(df["Sentiment"].value_counts())

# =========================
# About
# =========================
elif menu == "About":
    st.title("ℹ️ About Project")

    st.markdown("""
    ### NLP Sentiment Analysis Dashboard
    Dashboard ini digunakan untuk:
    - Menampilkan eksplorasi dataset review
    - Melakukan prediksi sentimen dari teks
    - Melakukan batch prediction melalui file CSV
    - Membandingkan performa beberapa skenario model NLP

    ### Model yang digunakan
    - Decision Tree + Bag of Words
    - Decision Tree + N-Gram
    - Decision Tree + TF-IDF

    ### Catatan
    Preprocessing pada dashboard ini dibuat konsisten dengan preprocessing saat training model.
    """)
