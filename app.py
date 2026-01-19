import streamlit as st
import re
import pickle
import torch
from gensim.models import Word2Vec
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -------------------- SETUP --------------------
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# -------------------- PREPROCESS FUNCTION --------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^a-z\s]", "", text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# -------------------- LOAD MODELS --------------------
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("tfidf_model.pkl", "rb") as f:
    tfidf_model = pickle.load(f)

w2v_model = Word2Vec.load("word2vec.model")

with open("word2vec_classifier.pkl", "rb") as f:
    w2v_classifier = pickle.load(f)

bert_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# -------------------- STREAMLIT UI --------------------
st.title("Customer Feedback Sentiment Analysis")

model_choice = st.selectbox(
    "Choose NLP Model",
    ["TF-IDF", "Word2Vec", "BERT"]
)

user_text = st.text_area("Enter customer feedback")

# -------------------- PREDICTION --------------------
if st.button("Analyze Sentiment"):

    if model_choice == "TF-IDF":
        clean_text = preprocess_text(user_text)
        vec = tfidf_vectorizer.transform([clean_text])
        pred = tfidf_model.predict(vec)
        st.success("Positive 😊" if pred[0] == 1 else "Negative 😠")

    elif model_choice == "Word2Vec":
        def sentence_vector(sentence):
            words = sentence.split()
            vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
            return sum(vectors) / len(vectors) if vectors else [0]*100

        clean_text = preprocess_text(user_text)
        vec = sentence_vector(clean_text)
        pred = w2v_classifier.predict([vec])
        st.success("Positive 😊" if pred[0] == 1 else "Negative 😠")

    else:
        result = bert_model(user_text)[0]
        st.success(f"{result['label']} ({round(result['score'], 2)})")
