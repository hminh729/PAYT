import pickle
import numpy as np
import nltk
from pre_data import clean_text

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Sigmoid cho output
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward pass MLP 1 hidden layer
def predict(X, W1, b1, W2, b2):
    # Kiểm tra shape
    if X.shape[1] != W1.shape[1]:
        raise ValueError(f"Input size mismatch: X has {X.shape[1]} features, W1 expects {W1.shape[1]}")

    # Forward pass
    Z1 = X @ W1.T + b1
    A1 = np.tanh(Z1)
    Z2 = A1 @ W2.T + b2
    A2 = sigmoid(Z2)

    # Chuyển sang 0/1
    return (A2 > 0.5).astype(int)

def main():
    # Load model đã train (trọng số + vectorizer)
    with open('mlp_spam_model.pkl', 'rb') as f:
        model_data = pickle.load(f)

    W1 = model_data['W1']
    b1 = model_data['b1']
    W2 = model_data['W2']
    b2 = model_data['b2']
    vectorizer = model_data['vectorizer']  # vectorizer đã fit lúc train

    # Ví dụ email cần dự đoán
    # email_text = "Hi Minh, please review the attached report before our meeting tomorrow."
    email_text = "Congratulations! You've won a $1000 gift card. Click here to claim now!"

    email_clean = clean_text(email_text)

    # Transform email bằng vectorizer đã fit
    X_vector = vectorizer.transform([email_clean]).toarray()
    # Dự đoán
    prediction = predict(X_vector, W1, b1, W2, b2)

    # In kết quả
    print("Spam email" if prediction[0][0] == 1 else "Not spam email")

if __name__ == "__main__":
    main()
