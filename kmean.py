import numpy as np
from sklearn.cluster import KMeans

# Assume embeddings_np is the array of embeddings from YAMNet
# embeddings_np shape: [Time, 1024]
embeddings_np = np.load("cityofstars.npy")


# Set the number of clusters (k)
num_clusters = 6  # You can adjust this based on your needs

# Perform k-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)


# Function to remove outliers using IQR
def remove_outliers(data):
    mean = np.mean(data.flatten(), axis=0)
    std = np.std(data.flatten(), axis=0)

    # Define the threshold for outliers (3 standard deviations from the mean)
    threshold_upper = mean + 1 * std
    threshold_lower = mean - 1 * std

    # Create a mask to filter out the outliers
    mask = (embeddings_np >= threshold_lower) & (embeddings_np <= threshold_upper)
    row_mask = np.all(mask, axis=1)
    # Apply the mask to filter the embeddings
    filtered_embeddings_np = np.where(mask, embeddings_np, 0)

    return filtered_embeddings_np


# Remove outliers
clean_data = remove_outliers(embeddings_np)
print(clean_data)
kmeans.fit(clean_data)

# Cluster labels for each embedding
cluster_labels = kmeans.labels_

# Cluster centers
cluster_centers = kmeans.cluster_centers_


# Print some results
print(f"Cluster Labels: {cluster_labels}")
print(f"Cluster Centers: {cluster_centers.shape}")  # (num_clusters, 1024)

new_embeddings = np.zeros((clean_data.shape[0], num_clusters))
for i, embedding in enumerate(clean_data):
    distances = np.linalg.norm(cluster_centers - embedding, axis=1)
    new_embeddings[i] = distances

np.save("reduced_embeddings_xyz.npy", new_embeddings)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(clean_data)

# Plot the clusters
plt.figure(figsize=(8, 6))
for cluster_id in range(num_clusters):
    cluster_points = reduced_embeddings[cluster_labels == cluster_id]
    plt.scatter(
        cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id}"
    )

plt.title("K-Means Clustering of Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()
