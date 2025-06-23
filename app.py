from flask import Flask, request, jsonify
import pickle, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt'); nltk.download('stopwords')

# --- LOAD ARTIFACTS ---
model  = pickle.load(open("model_svm_pso.pkl",    "rb"))
tfidf  = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
labels = pickle.load(open("label_encoder.pkl",    "rb")).classes_.tolist()  # ['negative','neutral','positive']

# --- PRE-PROCESS ---
stop_words, stemmer = set(stopwords.words("english")), PorterStemmer()
def clean_text(txt):
    tokens = [stemmer.stem(w) for w in word_tokenize(txt.lower()) if w.isalpha() and w not in stop_words]
    return " ".join(tokens)

# --- FLASK ---
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data   = request.get_json(force=True)
    review = data.get("text", "")
    vec    = tfidf.transform([clean_text(review)])
    pred   = model.predict(vec)[0]
    return jsonify({"sentiment": labels[pred]})

@app.route("/")
def home(): return "Sentiment Analysis API is running."

if __name__ == "__main__":
    app.run(debug=True)
