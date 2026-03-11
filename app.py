import streamlit as st
import pickle
import string

# Page Configuration 
st.set_page_config(
    page_title="Language Detection",
    page_icon="🌍",
    layout="centered",
)

# CSS Styling 
st.markdown("""
<style>
    .stApp {
        background-color: #f0f4f9;
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1a1a2e;
        text-align: center;
        padding: 10px 0 4px;
    }
    .subtitle {
        font-size: 1rem;
        color: #4a4a6a;
        text-align: center;
        margin-bottom: 30px;
    }
    .result-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        margin-top: 20px;
    }
    .result-label {
        font-size: 0.95rem;
        color: #a0aec0;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .result-language {
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff;
        margin: 8px 0 0;
    }
    .model-badge {
        display: inline-block;
        background: #2d6a4f;
        color: #ffffff;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 3px 12px;
        border-radius: 20px;
        margin-top: 10px;
    }
    .info-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 14px;
        font-size: 0.93rem;
        color: #374151;
    }
    .footer {
        text-align: center;
        font-size: 0.78rem;
        color: #9ca3af;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

#  Load Model and Vectorizer 
@st.cache_resource
def load_model():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

try:
    tfidf_vectorizer, svm_model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Text Cleaning (same as notebook) 
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text

# ── Prediction Function 
def predict_language(input_text):
    cleaned   = clean_text(input_text)
    vector    = tfidf_vectorizer.transform([cleaned])
    predicted = svm_model.predict(vector)[0]
    return predicted

# ── Title
st.markdown('<div class="main-title">🌍 Language Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by SVM — Highest Test Accuracy (98.6%)</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Error if .pkl files missing 
if not model_loaded:
    st.error(
        "Model files not found.\n\n"
        "Run the Jupyter notebook fully and execute the last cell "
        "to save tfidf_vectorizer.pkl and svm_model.pkl, "
        "then place both files in the same folder as app.py"
    )
    st.stop()

# ── Text Input 
st.markdown("#### ✏️ Enter a sentence in any language:")

user_input = st.text_area(
    label="",
    placeholder="e.g.  Bonjour, comment ça va ?",
    height=140,
    label_visibility="collapsed",
)

# ── Detect Button and Result 
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔍 Detect Language", use_container_width=True)

if predict_btn:
    if user_input.strip() == "":
        st.warning("Please enter a sentence before clicking Detect.")
    else:
        result = predict_language(user_input)
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Detected Language</div>
            <div class="result-language">{result}</div>
            <div class="model-badge">SVM · LinearSVC · Char n-gram TF-IDF</div>
        </div>
        """, unsafe_allow_html=True)

# ── Why SVM Section 
st.markdown("---")
st.markdown("####  Why SVM was chosen")

st.markdown("""
<div class="info-card">
    <b>Highest Test Accuracy — 98.6%</b><br>
    SVM achieved the best performance on unseen data among all four models tested.
</div>
<div class="info-card">
    <b>Character n-gram TF-IDF features</b><br>
    The vectorizer captures script-level and morphological patterns
    (unigrams to trigrams), making it extremely effective for multi-script language identification.
</div>
<div class="info-card">
    <b> LinearSVC — optimal for text</b><br>
    Linear kernel SVMs find the best separating hyperplane in high-dimensional
    TF-IDF feature space, which is ideal for text classification tasks.
</div>
""", unsafe_allow_html=True)

# ── Accuracy Comparison 
st.markdown("#### All Models — Accuracy Comparison")

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Naive Bayes",          "94.3%")
col_b.metric("Logistic Regression",  "98.2%")
col_c.metric("SVM ✅",               "98.6%", delta="Best")
col_d.metric("Random Forest",        "92.8%")

# ── Supported Languages 
st.markdown("---")
st.markdown("#### 🗺️ 22 Supported Languages")

languages = [
    "Arabic",    "Chinese",   "Dutch",     "English",  "Estonian",
    "French",    "Hindi",     "Indonesian","Japanese", "Korean",
    "Latin",     "Persian",   "Portugese", "Pushto",   "Romanian",
    "Russian",   "Spanish",   "Swedish",   "Tamil",    "Thai",
    "Turkish",   "Urdu",
]

cols = st.columns(5)
for idx, lang in enumerate(languages):
    cols[idx % 5].markdown(f"• {lang}")

