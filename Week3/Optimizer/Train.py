import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from CreateModel import InitModel
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.utils import to_categorical
import pickle

# Load data
path = "E:\PTIT\PAYT\Week3\Data\IMDB Dataset.csv"
data = pd.read_csv(path)
X = data["review"]
y = data["sentiment"]
le = LabelEncoder()
tfidf = TfidfVectorizer(max_features= 10000, stop_words="english")
y_vec = le.fit_transform(y)
y_vec = to_categorical(y_vec, num_classes = 2)
x_vec = tfidf.fit_transform(X).toarray()
x_train, x_test, y_train, y_test = train_test_split(x_vec, y_vec, test_size=0.2, random_state=42)
# Create Model
hidden_size = [64, 64]
model = InitModel(input_size=10000, output_size=2, hidden_size=hidden_size)

# Training loop

n_epoch = 10
learning_rate = 0.01
batch_size = 64

print("AdamW")
for epoch in range(n_epoch):
    epoch_loss = []
    epoch_acc = []

    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i: i + batch_size]
        y_batch = y_train[i: i + batch_size]

        forward_value = model.forward(x_batch)
        loss, acc = model.backward(x_batch, y_batch, learning_rate)
        epoch_loss.append(loss)
        epoch_acc.append(acc)

    print(f"Epoch {epoch + 1} / {n_epoch} - Loss: {np.mean(epoch_loss):.4f} - Acc: {np.mean(epoch_acc):.4f}")


test_predict = model.forward(x_test)
test_acc = accuracy_score(np.argmax(y_test, axis = 1), np.argmax(test_predict, axis = 1))
print(f"Test Accuracy: {test_acc:4f}")

with open("E:\PTIT\PAYT\Week3\Model.pkl", "wb") as f:
    pickle.dump(model, f)
