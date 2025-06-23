
from flask import Flask, request, jsonify
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data if not already present
nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
with open("model_svm_pso.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Define preprocessing
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

# Create Flask app
app = Flask(__name__)

# Sentiment labels (assumes 0=negative, 1=neutral, 2=positive)
labels = ['negative', 'neutral', 'positive']

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data.get("text", "")
    clean_review = clean_text(review)
    tfidf_input = tfidf.transform([clean_review])
    prediction = model.predict(tfidf_input)[0]
    sentiment = labels[prediction]
    return jsonify({"sentiment": sentiment})

@app.route("/", methods=["GET"])
def home():
    return "Sentiment Analysis API is running."

if __name__ == "__main__":
    app.run(debug=True)
