import streamlit as st
import pandas as pd
import joblib
import re
import html
import nltk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="NLP Sentiment Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #111111; color: #e0ddd8; }

section[data-testid="stSidebar"] {
    background-color: #1a1a1a !important;
    border-right: 1px solid #2e2e2e;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #c8c5c0 !important;
    font-size: 13px;
}

.sidebar-title {
    font-size: 11px; font-weight: 700; letter-spacing: 0.10em;
    text-transform: uppercase; color: #7a1c2e !important;
    padding-bottom: 8px; border-bottom: 1px solid #2e2e2e; margin-bottom: 16px;
}
.section-label {
    font-size: 11px; font-weight: 700; letter-spacing: 0.10em;
    text-transform: uppercase; color: #7a7875; margin-bottom: 8px;
}
.helper-text { font-size: 11px; color: #5a5855; margin-top: 4px; }

div[data-testid="metric-container"] {
    background-color: #1a1a1a; border: 1px solid #2e2e2e;
    border-radius: 8px; padding: 16px 20px !important;
}
div[data-testid="metric-container"] label {
    color: #7a7875 !important; font-size: 11px !important;
    font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #e0ddd8 !important; font-size: 26px !important; font-weight: 700;
}

h1, h2, h3 { color: #e8e5e0 !important; font-weight: 700; }
h1 { font-size: 20px !important; border-bottom: 2px solid #7a1c2e; padding-bottom: 8px; margin-bottom: 20px; }
h2 { font-size: 14px !important; color: #c8c5c0 !important; margin-top: 20px; }

.stButton > button {
    background-color: #7a1c2e !important; color: #f5f3f0 !important;
    border: none !important; border-radius: 6px !important;
    padding: 8px 20px !important; font-size: 13px !important;
    font-weight: 600 !important;
}
.stButton > button:hover { background-color: #9b2440 !important; }

.stTextArea > div > textarea,
.stTextInput > div > input {
    background-color: #1e1e1e !important; border: 1px solid #2e2e2e !important;
    border-radius: 6px !important; color: #e0ddd8 !important; font-size: 13px !important;
}
.stTextArea > div > textarea:focus,
.stTextInput > div > input:focus {
    border-color: #7a1c2e !important;
    box-shadow: 0 0 0 2px rgba(122,28,46,0.25) !important;
}
.stSelectbox > div > div {
    background-color: #1e1e1e !important; border: 1px solid #2e2e2e !important;
    border-radius: 6px !important; color: #e0ddd8 !important;
}

.result-box {
    background-color: #1e1e1e; border: 1px solid #2e2e2e;
    border-left: 4px solid #7a1c2e; border-radius: 8px;
    padding: 16px 20px; margin-top: 16px;
}
.result-box p { margin: 6px 0; font-size: 13px; color: #c8c5c0; }
.result-box span.label { color: #e0ddd8; font-weight: 600; }

.badge-positive { background:#1a3a1a; color:#6daa45; border:1px solid #3a5a2a; padding:3px 12px; border-radius:20px; font-size:13px; font-weight:600; }
.badge-neutral  { background:#3a3010; color:#e8af34; border:1px solid #5a4a10; padding:3px 12px; border-radius:20px; font-size:13px; font-weight:600; }
.badge-negative { background:#3a1a1a; color:#dd6974; border:1px solid #5a2a2a; padding:3px 12px; border-radius:20px; font-size:13px; font-weight:600; }

hr { border-color: #2e2e2e !important; }
.block-container { padding: 28px 32px 40px 32px !important; }

.stTabs [data-baseweb="tab-list"] { background-color: #1a1a1a; border-radius: 8px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #7a7875 !important; font-size: 13px !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] { background-color: #7a1c2e !important; color: #f5f3f0 !important; border-radius: 6px !important; }
</style>
""", unsafe_allow_html=True)


# ─── NLTK SETUP ───────────────────────────────────────────────────────────────
@st.cache_resource
def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    return set(stopwords.words('english')), WordNetLemmatizer()

stop_words, lemmatizer = setup_nltk()


# ─── PREPROCESSING ────────────────────────────────────────────────────────────
def preprocess_text(text):
    text = html.unescape(str(text))
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)


# ─── LOAD ARTIFACTS ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        "DT + BoW": {
            "vectorizer": joblib.load("models/vec_bow.joblib"),
            "model":      joblib.load("models/dt_bow.joblib"),
        },
        "DT + N-Gram": {
            "vectorizer": joblib.load("models/vec_ngram.joblib"),
            "model":      joblib.load("models/dt_ngram.joblib"),
        },
        "DT + TF-IDF": {
            "vectorizer": joblib.load("models/vec_tfidf.joblib"),
            "model":      joblib.load("models/dt_tfidf.joblib"),
        },
    }

@st.cache_data
def load_dataset():
    return pd.read_csv("reviews_sample.csv")

@st.cache_data
def load_results():
    return pd.read_csv("models/model_results.csv")

models    = load_models()
df        = load_dataset()
result_df = load_results()


# ─── HELPER: matplotlib dark style ────────────────────────────────────────────
def dark_fig(w=6, h=3.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors='#c8c5c0')
    for sp in ax.spines.values():
        sp.set_edgecolor('#2e2e2e')
    ax.yaxis.grid(True, color='#252525', linewidth=0.7)
    ax.set_axisbelow(True)
    return fig, ax


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">NLP Dashboard</p>', unsafe_allow_html=True)

    menu = st.radio(
        "Navigation",
        ["Overview", "Single Prediction", "Batch Prediction",
         "Model Performance", "Dataset Explorer", "About"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    selected_model = st.selectbox("Active Model", list(models.keys()))

    best_row = result_df.loc[result_df["Accuracy"].idxmax()]
    st.markdown(f"""
    <div style="margin-top:16px;padding:12px;background:#1e1e1e;border-radius:8px;border:1px solid #2e2e2e;">
        <p class="section-label" style="margin-bottom:6px;">Best Model</p>
        <p style="color:#e0ddd8;font-weight:700;font-size:13px;margin:0;">{best_row['Model']}</p>
        <p style="color:#7a7875;font-size:12px;margin:2px 0 0 0;">Accuracy: {best_row['Accuracy']:.4f}</p>
    </div>
    """, unsafe_allow_html=True)


# ─── OVERVIEW ─────────────────────────────────────────────────────────────────
if menu == "Overview":
    st.markdown("# NLP Sentiment Analysis")
    st.markdown('<p class="helper-text">Amazon Fine Food Reviews — Decision Tree + BoW / N-Gram / TF-IDF</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples",  f"{len(df):,}")
    c2.metric("Train Split",    "80%")
    c3.metric("Test Split",     "20%")
    c4.metric("Model Variants", "3")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("## Distribusi Sentimen")
        if "Sentiment" in df.columns:
            counts = df["Sentiment"].value_counts()
            fig, ax = dark_fig(5, 3.5)
            ax.bar(counts.index, counts.values,
                   color=["#6daa45","#e8af34","#dd6974"], width=0.45, edgecolor="none")
            for i, (idx, val) in enumerate(counts.items()):
                ax.text(i, val + 150, f"{val:,}", ha='center', color='#b0ada8',
                        fontsize=10, fontfamily='monospace')
            ax.set_ylabel("Count", color='#7a7875', fontsize=11)
            ax.tick_params(axis='x', colors='#c8c5c0', labelsize=11)
            ax.tick_params(axis='y', colors='#7a7875', labelsize=9)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("## Performa Model")
        styled = result_df.copy()
        styled["Accuracy"] = styled["Accuracy"].map("{:.4f}".format)
        st.dataframe(styled, use_container_width=True, hide_index=True)

        fig2, ax2 = dark_fig(5, 2.8)
        bar_colors = ["#7a1c2e" if m == best_row["Model"] else "#3a2a2e"
                      for m in result_df["Model"]]
        ax2.barh(result_df["Model"], result_df["Accuracy"],
                 color=bar_colors, edgecolor="none", height=0.35)
        for i, val in enumerate(result_df["Accuracy"]):
            ax2.text(val + 0.002, i, f"{val:.4f}", va='center',
                     color='#c8c5c0', fontsize=10, fontfamily='monospace')
        ax2.set_xlim(0, result_df["Accuracy"].max() * 1.15)
        ax2.tick_params(axis='y', colors='#c8c5c0', labelsize=11)
        ax2.tick_params(axis='x', colors='#7a7875', labelsize=9)
        ax2.xaxis.grid(True, color='#252525', linewidth=0.7)
        plt.tight_layout()
        st.pyplot(fig2); plt.close()


# ─── SINGLE PREDICTION ────────────────────────────────────────────────────────
elif menu == "Single Prediction":
    st.markdown("# Single Prediction")
    st.markdown(f'<p class="helper-text">Model aktif: {selected_model}</p>', unsafe_allow_html=True)

    user_input = st.text_area("Input teks review", height=150,
                              placeholder="This product is amazing and works really well...")

    if st.button("Run Prediction"):
        if not user_input.strip():
            st.warning("Masukkan teks terlebih dahulu.")
        else:
            clean  = preprocess_text(user_input)
            vec    = models[selected_model]["vectorizer"]
            clf    = models[selected_model]["model"]
            result = clf.predict(vec.transform([clean]))[0]

            badge_map = {
                "Positive": '<span class="badge-positive">Positive</span>',
                "Neutral":  '<span class="badge-neutral">Neutral</span>',
                "Negative": '<span class="badge-negative">Negative</span>',
            }
            st.markdown(f"""
            <div class="result-box">
                <p><span class="label">Model</span> &mdash; {selected_model}</p>
                <p><span class="label">Preprocessed</span> &mdash; <em>{clean[:120]}{"..." if len(clean)>120 else ""}</em></p>
                <p style="margin-top:10px;"><span class="label">Sentiment</span> &mdash; {badge_map.get(result, result)}</p>
            </div>
            """, unsafe_allow_html=True)


# ─── BATCH PREDICTION ─────────────────────────────────────────────────────────
elif menu == "Batch Prediction":
    st.markdown("# Batch Prediction")
    st.markdown('<p class="helper-text">Upload CSV, pilih kolom teks, jalankan prediksi massal.</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.markdown("**Preview**")
        st.dataframe(batch_df.head(5), use_container_width=True, hide_index=True)

        text_col = st.selectbox("Kolom teks", batch_df.columns)

        if st.button("Run Batch Prediction"):
            with st.spinner("Memproses..."):
                batch_df["clean_text"]          = batch_df[text_col].astype(str).apply(preprocess_text)
                vec                             = models[selected_model]["vectorizer"]
                clf                             = models[selected_model]["model"]
                batch_df["Predicted_Sentiment"] = clf.predict(vec.transform(batch_df["clean_text"]))

            st.success(f"Selesai — {len(batch_df):,} baris diproses.")
            dist = batch_df["Predicted_Sentiment"].value_counts().reset_index()
            dist.columns = ["Sentiment", "Count"]
            st.dataframe(dist, use_container_width=True, hide_index=True)
            st.dataframe(batch_df[[text_col, "Predicted_Sentiment"]].head(30),
                         use_container_width=True, hide_index=True)

            st.download_button("Download Hasil (.csv)",
                               data=batch_df.to_csv(index=False).encode("utf-8"),
                               file_name="hasil_prediksi_sentimen.csv",
                               mime="text/csv")


# ─── MODEL PERFORMANCE ────────────────────────────────────────────────────────
elif menu == "Model Performance":
    st.markdown("# Model Performance")

    tab1, tab2 = st.tabs(["Accuracy Comparison", "Confusion Matrix"])

    with tab1:
        styled = result_df.copy()
        styled["Accuracy"] = styled["Accuracy"].map("{:.4f}".format)
        st.dataframe(styled, use_container_width=True, hide_index=True)

        fig, ax = dark_fig(7, 3)
        bar_colors = ["#7a1c2e" if m == best_row["Model"] else "#3a2a2e"
                      for m in result_df["Model"]]
        bars = ax.bar(result_df["Model"], result_df["Accuracy"],
                      color=bar_colors, width=0.4, edgecolor="none")
        for bar, val in zip(bars, result_df["Accuracy"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.4f}", ha='center', color='#c8c5c0',
                    fontsize=10, fontfamily='monospace')
        ax.set_ylim(0, result_df["Accuracy"].max() * 1.18)
        ax.set_ylabel("Accuracy", color='#7a7875', fontsize=11)
        ax.tick_params(axis='x', colors='#c8c5c0', labelsize=11)
        ax.tick_params(axis='y', colors='#7a7875', labelsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab2:
        st.markdown(f"**Confusion Matrix — {selected_model}**")
        if "Sentiment" in df.columns:
            X = df["clean_text"] if "clean_text" in df.columns else df["Text"].apply(preprocess_text)
            y = df["Sentiment"]
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

            vec    = models[selected_model]["vectorizer"]
            clf    = models[selected_model]["model"]
            y_pred = clf.predict(vec.transform(X_test))

            labels = ["Positive", "Neutral", "Negative"]
            cm     = confusion_matrix(y_test, y_pred, labels=labels)
            disp   = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            fig_cm.patch.set_facecolor("#1a1a1a")
            ax_cm.set_facecolor("#1a1a1a")
            disp.plot(ax=ax_cm, cmap="RdPu", colorbar=False)
            ax_cm.set_title(f"Confusion Matrix — {selected_model}",
                            color='#e0ddd8', fontsize=12, pad=12)
            ax_cm.tick_params(colors='#c8c5c0')
            ax_cm.set_xlabel("Predicted", color='#7a7875')
            ax_cm.set_ylabel("True", color='#7a7875')
            for sp in ax_cm.spines.values(): sp.set_edgecolor('#2e2e2e')
            plt.tight_layout()
            st.pyplot(fig_cm); plt.close()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy",  f"{accuracy_score(y_test, y_pred):.4f}")
            m2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
            m3.metric("Recall",    f"{recall_score(y_test, y_pred,    average='weighted', zero_division=0):.4f}")
            m4.metric("F1-Score",  f"{f1_score(y_test, y_pred,        average='weighted', zero_division=0):.4f}")


# ─── DATASET EXPLORER ─────────────────────────────────────────────────────────
elif menu == "Dataset Explorer":
    st.markdown("# Dataset Explorer")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows",    f"{len(df):,}")
    c2.metric("Total Columns", df.shape[1])
    if "Sentiment" in df.columns:
        c3.metric("Unique Labels", df["Sentiment"].nunique())

    st.markdown("## Preview (20 rows)")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    st.markdown("## Schema")
    st.dataframe(pd.DataFrame({
        "Column":    df.columns,
        "Non-Null":  df.notnull().sum().values,
        "Dtype":     df.dtypes.values.astype(str)
    }), use_container_width=True, hide_index=True)

    if "Sentiment" in df.columns:
        st.markdown("## Label Distribution")
        dist = df["Sentiment"].value_counts().reset_index()
        dist.columns = ["Sentiment", "Count"]
        dist["Proportion"] = (dist["Count"] / dist["Count"].sum()).map("{:.2%}".format)
        st.dataframe(dist, use_container_width=True, hide_index=True)


# ─── ABOUT ────────────────────────────────────────────────────────────────────
elif menu == "About":
    st.markdown("# About")
    st.markdown("""
    <div style="background:#1a1a1a;border:1px solid #2e2e2e;border-radius:8px;padding:24px;margin-top:8px;">
        <p style="font-size:13px;color:#b0ada8;line-height:1.7;">
            Dashboard ini merupakan bagian dari tugas mata kuliah
            <strong style="color:#e0ddd8;">Natural Language Processing</strong>
            yang bertujuan menganalisis sentimen pada dataset Amazon Fine Food Reviews.
        </p>
        <hr style="border-color:#2e2e2e;margin:16px 0;">
        <p class="section-label">Model Eksperimen</p>
        <table style="width:100%;border-collapse:collapse;font-size:13px;color:#b0ada8;">
            <thead>
                <tr>
                    <th style="text-align:left;padding:8px 12px;border-bottom:1px solid #2e2e2e;color:#7a7875;font-size:11px;text-transform:uppercase;letter-spacing:0.08em;">Model</th>
                    <th style="text-align:left;padding:8px 12px;border-bottom:1px solid #2e2e2e;color:#7a7875;font-size:11px;text-transform:uppercase;letter-spacing:0.08em;">Vectorizer</th>
                    <th style="text-align:left;padding:8px 12px;border-bottom:1px solid #2e2e2e;color:#7a7875;font-size:11px;text-transform:uppercase;letter-spacing:0.08em;">Classifier</th>
                </tr>
            </thead>
            <tbody>
                <tr><td style="padding:8px 12px;border-bottom:1px solid #252525;">Eksperimen 1</td><td style="padding:8px 12px;border-bottom:1px solid #252525;">Bag of Words</td><td style="padding:8px 12px;border-bottom:1px solid #252525;">Decision Tree</td></tr>
                <tr><td style="padding:8px 12px;border-bottom:1px solid #252525;">Eksperimen 2</td><td style="padding:8px 12px;border-bottom:1px solid #252525;">N-Gram (Bigram)</td><td style="padding:8px 12px;border-bottom:1px solid #252525;">Decision Tree</td></tr>
                <tr><td style="padding:8px 12px;">Eksperimen 3</td><td style="padding:8px 12px;">TF-IDF</td><td style="padding:8px 12px;">Decision Tree</td></tr>
            </tbody>
        </table>
        <hr style="border-color:#2e2e2e;margin:16px 0;">
        <p class="section-label">Preprocessing Pipeline</p>
        <p style="font-size:13px;color:#b0ada8;line-height:1.7;">
            HTML unescape &rarr; Lowercase &rarr; Remove URL &rarr; Remove non-alpha &rarr; Tokenize &rarr; Lemmatize &rarr; Remove stopwords
        </p>
        <hr style="border-color:#2e2e2e;margin:16px 0;">
        <p class="section-label">Dataset</p>
        <p style="font-size:13px;color:#b0ada8;">Amazon Fine Food Reviews &mdash; 50,000 samples (stratified split)</p>
    </div>
    """, unsafe_allow_html=True)
