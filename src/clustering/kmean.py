import numpy as np
from sklearn.cluster import KMeans
import argparse

parser = argparse.ArgumentParser(
    prog="Kmean Cluster", description="This program will cluster the given .npy file"
)

parser.add_argument(
    "--clusters", type=int, help="THe number of clusters", required=True
)
parser.add_argument(
    "--filepath", type=str, help="The path to the .npy file", required=True
)
parser.add_argument(
    "--outputpath",
    type=str,
    help="The path for the output file",
    default="reduced_embeddings.npy",
)
parser.add_argument(
    "--showvisuals",
    type=bool,
    help="If enabled shows a visualization of the resulting clusters",
    default=False,
    required=False,
)
args = parser.parse_args()

# Assume embeddings_np is the array of embeddings from YAMNet
# embeddings_np shape: [Time, 1024]
embeddings_np = np.load(args.filepath)


# Perform k-means clustering
kmeans = KMeans(n_clusters=args.clusters, random_state=42)


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
kmeans.fit(clean_data)

# Cluster labels for each embedding
cluster_labels = kmeans.labels_

# Cluster centers
cluster_centers = kmeans.cluster_centers_


# Print some results
print(f"Cluster Labels: {cluster_labels}")
print(f"Cluster Centers: {cluster_centers.shape}")  # (args.clusters, 1024)

new_embeddings = np.zeros((clean_data.shape[0], args.clusters))
for i, embedding in enumerate(clean_data):
    distances = np.linalg.norm(cluster_centers - embedding, axis=1)
    new_embeddings[i] = distances

np.save(
    args.outputpath if ".npy" in args.outputpath else args.outputpath + ".npy",
    new_embeddings,
)


if not args.showvisuals:
    exit()


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(clean_data)

# Plot the clusters
plt.figure(figsize=(8, 6))
for cluster_id in range(args.clusters):
    cluster_points = reduced_embeddings[cluster_labels == cluster_id]
    plt.scatter(
        cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id}"
    )

plt.title("K-Means Clustering of Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()
