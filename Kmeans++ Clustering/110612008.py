import numpy as np

def distance(x1, x2):
    return np.sqrt(np.sum(np.power(x1 - x2, 2)))

def initialize_centroids(X, k):
    m, n = np.shape(X)
    centroids = []
    centroids.append(X[np.random.choice(m)])
    for i in range(1, k):
        distances = np.array([min(distance(x, centroid) ** 2 for centroid in centroids) for x in X])
        probabilities = distances / distances.sum()
        next_centroid = X[np.random.choice(m, p = probabilities)]
        centroids.append(next_centroid)
    return np.array(centroids)

def assign_clusters(X, centroids):
    clusters = []
    for x in X:
        distances = [distance(x, centroid) for centroid in centroids]
        clusters.append(np.argmin(distances))
    return np.array(clusters)

def update_centroids(X, clusters, k):
    n = X.shape[1]
    centroids = np.zeros((k, n))
    for i in range(k):
        points = X[clusters == i]
        if len(points) > 0:
            centroids[i] = np.mean(points, axis = 0)
    return centroids
def kmeans(X, k, max_iterations = 1000):
    centroids = initialize_centroids(X, k)
    for i in range(max_iterations):
        clusters = assign_clusters(X, centroids)
        previous_centroids = centroids
        centroids = update_centroids(X, clusters, k)
        diff = np.sum([distance(previous_centroids[i], centroids[i]) for i in range(k)])
        if diff < 1e-6:
            break
    return clusters

def clustering(X, k = 13):
    y = kmeans(X, k);
    return y

if __name__ == "__main__":
    # load data
    X = np.load("./data.npy") # size: [10000, 512]
    y = clustering(X)
    # save clustered labels
    np.save("110612008.npy", y) # output size should be [10000]
    
