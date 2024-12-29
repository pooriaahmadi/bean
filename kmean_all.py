import numpy as np
from sklearn.cluster import KMeans
import os
import pickle

# Assume embeddings_np is the array of embeddings from YAMNet
# embeddings_np shape: [Time, 1024]
embeddings = np.zeros((0, 1024))
for file in os.listdir("embeddings"):
    array = np.load(f"embeddings/{file}")
    embeddings = np.vstack((embeddings, array))


# Set the number of clusters (k)
num_clusters = 3  # You can adjust this based on your needs

# Perform k-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)


# Function to remove outliers using IQR
def remove_outliers(data):
    mean = np.mean(data.flatten(), axis=0)
    std = np.std(data.flatten(), axis=0)

    # Define the threshold for outliers (3 standard deviations from the mean)
    threshold_upper = mean + 3 * std
    threshold_lower = mean - 3 * std

    # Create a mask to filter out the outliers
    mask = (data >= threshold_lower) & (data <= threshold_upper)
    # Apply the mask to filter the embeddings
    filtered_embeddings_np = np.where(mask, data, 0)

    return filtered_embeddings_np


# Remove outliers
# clean_data = remove_outliers(embeddings)
# print(clean_data)
kmeans.fit(embeddings)
clean_data = embeddings

# Cluster labels for each embedding
cluster_labels = kmeans.labels_

# Cluster centers
cluster_centers = kmeans.cluster_centers_

# Save the model
with open("kmeans_model.pkl", "wb") as file:
    pickle.dump(kmeans, file)

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
