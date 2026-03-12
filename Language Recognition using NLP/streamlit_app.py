import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("svm_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("🌐 Language Recognition using NLP")
st.write("Enter text below to detect its language:")

user_input = st.text_area("Input Text", height=150)

if st.button("Detect Language"):
    if user_input.strip():
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)
        st.success(f"Detected Language: **{prediction[0]}**")
    else:
        st.warning("Please enter some text.")