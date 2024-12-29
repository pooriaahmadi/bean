import pickle
import numpy as np

# Later, to load the model:
with open("kmeans_model.pkl", "rb") as file:
    kmeans = pickle.load(file)

# Cluster labels for each embedding
cluster_labels = kmeans.labels_

# Cluster centers
cluster_centers = kmeans.cluster_centers_
num_clusters = 3

# Print some results
print(f"Cluster Labels: {cluster_labels}")
print(f"Cluster Centers: {cluster_centers.shape}")  # (num_clusters, 1024)
music = np.load("cityofstars.npy")
new_embeddings = np.zeros((music.shape[0], num_clusters))
for i, embedding in enumerate(music):
    distances = np.linalg.norm(cluster_centers - embedding, axis=1)
    new_embeddings[i] = distances

np.save("reduced_embeddings_xyz.npy", new_embeddings)
