# Install packages (run ini di terminal, bukan di script)
# pip install pandas numpy scikit-learn matplotlib seaborn wordcloud nltk pyswarm

# ========== IMPORT LIBRARY ==========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from pyswarm import pso

# Download resource NLTK (hanya perlu sekali)
nltk.download('punkt')
nltk.download('stopwords')

# ========== LOAD DATASET ==========
csv_path = r"D:\masterprog\XAMPP\htdocs\Pendat_Project006\amazone_reviews.csv"
df = pd.read_csv(csv_path,
                 usecols=['reviews.text', 'reviews.rating', 'reviews.title', 'name'],
                 low_memory=False)
df.dropna(subset=['reviews.text', 'reviews.rating'], inplace=True)
df.columns = ['review', 'rating', 'title', 'product']  # rename kolom

# dibatasi 3000 data karena lemot kalo ngeload 34.660 baris (ulasan), dgn 21 kolom
df = df.sample(n=3000, random_state=42).reset_index(drop=True)
# ========== LABELING SENTIMEN ==========
def label_sentiment(rating):
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'

df['sentiment'] = df['rating'].apply(label_sentiment)

# ========== COMBINE REVIEW + TITLE (mastiin smuanya string dlu) ==========
df['title']  = df['title'].fillna('').astype(str)
df['review'] = df['review'].fillna('').astype(str)
df['full_text'] = (df['title'] + " " + df['review']).str.strip()

# ========== TEXT PREPROCES ==========
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

df['clean_review'] = df['full_text'].apply(clean_text)

# ========== TF-IDF ==========
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_review'])  # tetap sparse untuk efisiensi memori

# ========== LABEL ENCODING ==========
y = df['sentiment']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ========== SPLIT DATA (TRAIN - VALIDATION - TEST) ==========
X_temp, X_test, y_temp, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)  # 0.1765 × 85% ≈ 15%

# ========== PSO OBJECTIVE ==========
from sklearn.svm import SVC

def objective_function(params):
    C, gamma = params
    print(f"Evaluating C={C}, gamma={gamma}")
    model = SVC(C=C, gamma=gamma, kernel='rbf', class_weight='balanced')  # gunakan class_weight
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    score = accuracy_score(y_val, preds) #score = accuracy_score(y_val, preds)
    print(f"Validation Accuracy: {score}")
    return -score  # karena PSO meminimalkan fungsi

# ========== PSO HYPERPARAMETER TUNING ==========
lb = [0.1, 0.00001]
ub = [100, 1]
best_params, _ = pso(objective_function, lb, ub, swarmsize=10, maxiter=5)
C_opt, gamma_opt = best_params

# ========== TRAIN FINAL MODEL (TRAIN + VALIDATION) ==========
X_final_train = np.vstack((X_train.toarray(), X_val.toarray()))
y_final_train = np.concatenate((y_train, y_val))

model = SVC(C=C_opt, gamma=gamma_opt, kernel='rbf', class_weight='balanced')
model.fit(X_final_train, y_final_train)
y_pred = model.predict(X_test.toarray())

# ========== EVALUASI HASIL ==========
print("=== Classification Report ===") #untuk Precision, Recall, F1-Score untuk tiap kelas (positive, neutral, negative).
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ========== WORDCLOUD PER SENTIMEN ==========
for sentiment_label in le.classes_:
    text = " ".join(df[df['sentiment'] == sentiment_label]['clean_review'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(12, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud - {sentiment_label.capitalize()} Reviews")
    plt.show()

# ========== FREKUENSI KATA TERBANYAK (POSITIVE) ==========
from collections import Counter
text = " ".join(df[df['sentiment'] == 'positive']['clean_review'])
all_words = text.split()
freq_dist = Counter(all_words)
common_words = freq_dist.most_common(10)

# BAR CHART TOP 10
words, counts = zip(*common_words)
plt.figure(figsize=(10, 5))
sns.barplot(x=list(words), y=list(counts), palette='viridis')
plt.title("Top 10 Most Common Words in Positive Reviews")
plt.ylabel("Frequency")
plt.xlabel("Words")
plt.show()
