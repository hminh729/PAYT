import numpy as np
from pre_data import pre_data, clean_text
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score

# ReLU và Sigmoid
def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# Tính toán giá trị hàm loss
def cost(y_true, A):
    m = y_true.shape[0]
    return -(1/m) * np.sum(y_true * np.log(A + 1e-8) + (1 - y_true) * np.log(1 - A + 1e-8))

# Khởi tạo tham số
def init(X, hidden_size=50):
    d0 = X.shape[1]
    d1 = hidden_size
    d2 = 1  # output neuron
    W1 = np.random.randn(d1, d0) * np.sqrt(2 / d0)
    W2 = np.random.randn(d2, d1) * np.sqrt(2 / d1)
    b1 = np.zeros((d1, 1))
    b2 = np.zeros((d2, 1))
    return W1, b1, W2, b2

# Forward
def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X.T) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache = (Z1, A1, Z2, A2)
    return A2, cache

# Backward
def backward(X, y, W1, W2, cache):
    m = X.shape[0]
    Z1, A1, Z2, A2 = cache

    dZ2 = A2 - y.T  # (1, m)
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * (Z1 > 0)  # ReLU gradient
    dW1 = (1/m) * np.dot(dZ1, X)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

# Training
def train(X_train, y_train, hidden_size=50, learning_rate=0.2, epochs=1000):
    W1, b1, W2, b2 = init(X_train, hidden_size)
    losses = []

    for epoch in range(epochs):
        # Forward
        A2, cache = forward(X_train, W1, b1, W2, b2)
        loss = cost(y_train, A2.T)
        losses.append(loss)

        # Backward
        dW1, db1, dW2, db2 = backward(X_train, y_train, W1, W2, cache)

        # Update
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    return W1, b1, W2, b2, losses

# Prediction
def predict(X, W1, b1, W2, b2):
    A2, _ = forward(X, W1, b1, W2, b2)
    return (A2.flatten() > 0.5).astype(int)

# Main
# Main
if __name__ == "__main__":
    scaler = MaxAbsScaler()
    X, y, feature_names, vectorizer = pre_data('data/spam.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Scale dữ liệu
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Train MLP
    W1, b1, W2, b2, losses = train(X_train_scaled, y_train, hidden_size=100, learning_rate=0.1, epochs=1000)
    # Dự đoán trên tập test
    y_pred = predict(X_test_scaled, W1, b1, W2, b2)
    # Tính accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    # Lưu mô hình
    with open('mlp_spam_model.pkl', 'wb') as f:
        pickle.dump({
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2,
            'vectorizer': vectorizer
        }, f)



