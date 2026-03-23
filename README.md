# 🌐 Lingua_Detect_AI

A Natural Language Processing (NLP) project that detects the language of any input text using classical machine learning models. Four models are trained, evaluated, and compared — with the best-performing model (SVM) deployed as an interactive web app using **Streamlit**.

---

## 📌 What is this Project?

This project builds a **multi-class text classification system** that automatically identifies which language a given sentence or phrase belongs to. It uses character-level and word-level features extracted from text, trains multiple ML classifiers, and deploys the best model through a clean Streamlit interface.

---

## ✨ Features

- 🔤 **Multi-language detection** — identifies languages from a diverse dataset (`language.csv`)
- 🧹 **Text preprocessing** — lowercasing and punctuation removal for cleaner features
- 📊 **Rich feature engineering** — character count, word count, vowel/consonant ratio, punctuation density, and more
- 🤖 **Four ML models trained and compared:**
  - Multinomial Naive Bayes (Bag-of-Words)
  - Logistic Regression (TF-IDF character n-grams)
  - Support Vector Machine — LinearSVC (TF-IDF character n-grams)
  - Random Forest (TF-IDF word-level)
- 📈 **Full evaluation** — Accuracy, Precision, Recall, F1 Score, Classification Report, and Confusion Matrix for every model
- 📉 **Visual comparisons** — bar charts, pie charts, and grouped metric charts across all models
- 🏆 **Best model (SVM) saved** as `.pkl` files and deployed via Streamlit
- 🌐 **Interactive Streamlit app** — type any sentence and get an instant language prediction

---

## 🧠 Models & Results

| Model | Train Accuracy | Test Accuracy | Notes |
|---|---|---|---|
| Multinomial Naive Bayes | ~98.4% | ~94.3% | Fast baseline; moderate overfitting |
| Logistic Regression | ~98.4% | ~98.2% | Best generalisation; minimal train-test gap |
| **SVM (LinearSVC)** | **~99.7%** | **~98.6%** | **Best overall accuracy; deployed model** |
| Random Forest | ~95.1% | ~92.8% | Most stable; lower accuracy on word-level features |

> ✅ **SVM** was selected for deployment due to its highest test accuracy and strong generalisation using character-level TF-IDF n-gram features (unigrams to trigrams).

---

## 🗂️ Project Structure
```
├── language.csv                   # Dataset with text samples and language labels
├── Language_Recognition_NLP.ipynb # Full notebook: EDA, training, evaluation, model export
├── app.py                         # Streamlit web application
├── svm_model.pkl                  # Trained SVM model (saved via pickle)
├── tfidf_vectorizer.pkl           # Fitted TF-IDF vectorizer (saved via pickle)
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## ⚙️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.x |
| Data Handling | Pandas, NumPy |
| NLP & Features | Scikit-learn (CountVectorizer, TfidfVectorizer) |
| ML Models | Scikit-learn (MultinomialNB, LogisticRegression, LinearSVC, RandomForestClassifier) |
| Evaluation | Scikit-learn (accuracy_score, classification_report, confusion_matrix) |
| Visualisation | Matplotlib, Seaborn |
| Deployment | Streamlit |
| Model Persistence | Pickle |

---

## 🚀 Getting Started — Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/AbhishekDhawan07/Lingua_Detect_AI.git
cd Lingua_Detect_AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)

If you want to retrain and regenerate the `.pkl` files, run the notebook:
```bash
jupyter notebook Language_Recognition_NLP.ipynb
```

Run all cells — `svm_model.pkl` and `tfidf_vectorizer.pkl` will be saved in the project folder.

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🖥️ Streamlit App — How to Use

1. Open the app at `http://localhost:8501`
2. Type or paste any sentence in the text input box
3. Click **Detect Language**
4. The predicted language is displayed instantly

> The app uses the saved **SVM model** and **TF-IDF vectorizer** (`.pkl` files) — make sure both files are in the same directory as `app.py`.

---

## 📦 requirements.txt
```
numpy
pandas
matplotlib
seaborn
scikit-learn
streamlit
```

---

## 📊 Exploratory Data Analysis & Feature Engineering

The following features were engineered from raw text before model training:

| Feature | Description |
|---|---|
| `char_count` | Total number of characters |
| `word_count` | Total number of words |
| `avg_word_len` | Average word length |
| `digit_count` | Number of digit characters |
| `upper_count` | Number of uppercase characters |
| `lower_count` | Number of lowercase characters |
| `punct_count` | Number of punctuation marks |
| `unique_chars` | Number of unique characters |
| `vowel_count` | Number of vowels |
| `consonant_count` | Number of consonants |
| `space_count` | Number of spaces |
| `first_cap` | Whether the first character is uppercase |
| `ends_punct` | Whether the text ends with punctuation |

---

## 🏁 Conclusions

- **SVM** is the best-performing model (~98.6% test accuracy) leveraging character n-gram TF-IDF features that capture script and morphological patterns across languages.
- **Logistic Regression** is the most balanced model with the smallest gap between training and test accuracy, indicating minimal overfitting.
- **Naive Bayes** is a strong and fast baseline but shows slightly more overfitting due to word-level features and the independence assumption.
- **Random Forest** is the most stable model but has lower accuracy, as word-level TF-IDF with limited features doesn't capture language-distinguishing patterns as effectively as character n-grams.

---

## 🙌 Acknowledgements

- Dataset: `language.csv` — a labelled multilingual text dataset
- Libraries: Scikit-learn, Streamlit, Pandas, Matplotlib, Seaborn

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
