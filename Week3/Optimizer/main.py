import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def main():
    # Load model
    with open("E:/PTIT/PAYT/Week3/Model.pkl", "rb") as f:
        model = pickle.load(f)

    # Fit lại tfidf trên dataset gốc
    data = pd.read_csv("E:/PTIT/PAYT/Week3/Data/IMDB Dataset.csv")
    tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
    tfidf.fit(data["review"])

    while True:
        # Input review mới
        review = input("Nhập review: ")
        if review == "exit":
            break
        x_vec = tfidf.transform([review]).toarray()

        pred = model.forward(x_vec)
        label = np.argmax(pred, axis=1)[0]
        sentiment = "Positive 😀" if label == 1 else "Negative 😞"
        print(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
