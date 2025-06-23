# ======= 1. LIBRARIES =======
import pandas as pd, numpy as np, pickle, nltk, matplotlib.pyplot as plt, seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pyswarm import pso
from scipy import sparse
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

# ======= 2. LOAD & PREPROCESS =======
csv_path = r"D:\masterprog\XAMPP\htdocs\Pendat_Project006\amazone_reviews.csv"
df = (pd.read_csv(csv_path, usecols=['reviews.text', 'reviews.rating', 'reviews.title', 'name'], low_memory=False)
        .dropna(subset=['reviews.text', 'reviews.rating'])
        .sample(n=10000, random_state=42)
        .reset_index(drop=True))

df.columns = ['review', 'rating', 'title', 'product']

def label_sentiment(rating):
    return 'positive' if rating >= 4 else 'neutral' if rating == 3 else 'negative'

df['sentiment'] = df['rating'].apply(label_sentiment)
df['full_text'] = (df['title'].fillna('') + ' ' + df['review'].fillna('')).str.strip()

stop_words, stemmer = set(stopwords.words('english')), PorterStemmer()
def clean_text(t):
    t = t.lower()
    tokens = word_tokenize(t)
    tokens = [stemmer.stem(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(tokens)

df['clean_review'] = df['full_text'].apply(clean_text)

# ======= 3. TF-IDF =======
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_review'])
y = df['sentiment'].values
le = LabelEncoder().fit(y)
y_enc = le.transform(y)

# Save fitted artifacts
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

# ======= 4. SPLIT =======
X_temp, X_test, y_temp, y_test = train_test_split(X, y_enc, test_size=0.15, stratify=y_enc, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42)

# ======= 5. PSO TUNING =======
def objective(params):
    C, gamma = params
    clf = SVC(C=C, gamma=gamma, kernel='rbf', class_weight='balanced')
    clf.fit(X_train, y_train)
    return -accuracy_score(y_val, clf.predict(X_val))

best_params, _ = pso(objective, lb=[0.1, 1e-5], ub=[100, 1], swarmsize=10, maxiter=5, debug=False)
C_opt, gamma_opt = best_params
print(f"Best C={C_opt:.4f}, gamma={gamma_opt:.6f}")

# ======= 6. FINAL TRAINING =======
X_final = sparse.vstack([X_train, X_val])
y_final = np.concatenate([y_train, y_val])

model = SVC(C=C_opt, gamma=gamma_opt, kernel='rbf', class_weight='balanced')
model.fit(X_final, y_final)

# Save final model
pickle.dump(model, open("model_svm_pso.pkl", "wb"))

# ======= 7. EVALUASI =======
y_pred = model.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ======= 8. WORDCLOUD per SENTIMEN =======
for label in le.classes_:
    text = " ".join(df[df['sentiment'] == label]['clean_review'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud: {label}")
    plt.tight_layout()
    plt.show()

# ======= 9. TOP 10 FREQUENT WORDS (POSITIVE) =======
positive_words = " ".join(df[df['sentiment'] == 'positive']['clean_review']).split()
word_freq = Counter(positive_words).most_common(10)
words, freqs = zip(*word_freq)

plt.figure(figsize=(8, 5))
sns.barplot(x=list(words), y=list(freqs), palette="viridis")
plt.title("Top 10 Frequent Words in Positive Reviews")
plt.ylabel("Frequency"); plt.xlabel("Word")
plt.tight_layout()
plt.show()
