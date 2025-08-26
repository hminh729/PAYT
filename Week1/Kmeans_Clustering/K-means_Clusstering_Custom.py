import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage.transform import rescale

def load_img(path):
    return mpimg.imread(path)

def show_img(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def scale_img(img, k):
    return rescale(img, k, channel_axis=2, anti_aliasing=True)

def init_centroid(X, k):
    idx = np.random.choice(len(X), k, replace=False)
    return X[idx]

def eucli_distance_matrix(X, centroids):
    return np.sum((X[:, np.newaxis] - centroids) ** 2, axis=2)

def assign_clusters(X, centroids):
    distances = eucli_distance_matrix(X, centroids)
    labels = np.argmin(distances, axis=1)
    return labels

def update_centroids(X, labels, K):
    centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else np.zeros(X.shape[1]) for i in range(K)])
    return centroids

def check(old_centroids, new_centroids, tol=1e-4):
    return np.sum((old_centroids - new_centroids) ** 2) < tol

def kmeans(X, K, max_iters=100):
    centroids = init_centroid(X, K)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, K)
        if check(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

def recreate_image(centroids, labels, w, h):
    img = centroids[labels].reshape(w, h, -1)
    return img

if __name__ == "__main__":
    img = load_img("./Data/man.jpg")
    img = scale_img(img, 0.5)  # giảm scale để nhanh hơn
    w, h, c = img.shape
    X = img.reshape(w * h, c)
    K = 7
    centroids, labels = kmeans(X, K)
    img = scale_img(img,2 )
    new_img = recreate_image(centroids, labels, w, h)
    show_img(new_img)
