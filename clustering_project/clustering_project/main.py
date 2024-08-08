from clustering_project.Clustering.FLKmeans import KMeans
import numpy as np 
from clustering_project.Clustering.Distances import DistanceMetric

# Example usage
"""data = np.array([
    [1, 0],
    [0, 1],
    [0, 0],
    [1, 1],
    [1, 1]
])

# Create a KMeans object with Jaccard distance
kmeans = KMeans(n_clusters=2, distance_metric=DistanceMetric.JACCARD)
labels = kmeans.fit(data)

print("Cluster labels:", labels)
print("Centroids:", kmeans.centroids)

# Plot the clusters
kmeans.plot_clusters(data)
"""