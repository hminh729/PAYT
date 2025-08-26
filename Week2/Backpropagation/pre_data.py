import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import re

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def convert_text_to_vector(text, vectorizer=None): # Chuyen text thanh vector
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        X_sparse = vectorizer.fit_transform(text)
    else:
        X_sparse = vectorizer.transform(text)
    return X_sparse.toarray(), vectorizer.get_feature_names_out(), vectorizer


def pre_data(path, vectorizer=None):
    df = pd.read_csv(path, encoding="latin-1")
    df = df[['v1','v2']]
    df.columns = ['label', 'message']

    df['label'] = df['label'].map({'ham': 0, 'spam': 1}) # Gán nhãn cho label

    df['message'] = df['message'].apply(clean_text) # Làm sạch message
    # Chuyển message thành vector
    X, feature_names, vectorizer = convert_text_to_vector(df['message'], vectorizer)

    y = df['label'].values

    return X, y, feature_names, vectorizer
