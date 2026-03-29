import streamlit as st
import pickle
import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from transformers import pipeline

# Load models
tfidf_model = pickle.load(open('tfidf_model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Load BERT
bert_model = pipeline("sentiment-analysis")

# Preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^a-z\s]", "", text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# Streamlit UI
st.title("Customer Feedback Sentiment Analysis")

text_input = st.text_area("Enter your text")

model_choice = st.selectbox("Choose Model", ["TF-IDF", "BERT"])

if st.button("Analyze"):
    if text_input:

        if model_choice == "TF-IDF":
            clean_text = preprocess_text(text_input)
            vector = tfidf_vectorizer.transform([clean_text])
            prediction = tfidf_model.predict(vector)[0]

            if prediction == 1:
                st.success("Positive Sentiment 😊")
            else:
                st.error("Negative Sentiment 😞")

        elif model_choice == "BERT":
            result = bert_model(text_input)[0]

            if result['label'] == 'POSITIVE':
                st.success("Positive Sentiment 😊")
            else:
                st.error("Negative Sentiment 😞")

    else:
        st.warning("Please enter some text")