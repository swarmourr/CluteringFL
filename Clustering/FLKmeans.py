import numpy as np
import matplotlib.pyplot as plt
from .Distances import DistanceMetric


class KMeans:
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-4, distance_metric=DistanceMetric.EUCLIDEAN, p=3):
        """
        KMeans clustering class supporting multiple distance metrics.

        Parameters:
        - n_clusters: int, default=2
            The number of clusters to form.
        - max_iter: int, default=100
            Maximum number of iterations of the k-means algorithm for a single run.
        - tol: float, default=1e-4
            Tolerance to declare convergence.
        - distance_metric: DistanceMetric, default=DistanceMetric.EUCLIDEAN
            The distance metric to use. Options: DistanceMetric.EUCLIDEAN, DistanceMetric.COSINE,
            DistanceMetric.MANHATTAN, DistanceMetric.MINKOWSKI, DistanceMetric.JACCARD.
        - p: int, default=3
            The Minkowski distance parameter. Used only if distance_metric is DistanceMetric.MINKOWSKI.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.distance_metric = distance_metric
        self.p = p
        self.centroids = None
        self.labels_ = None

    def _euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def _cosine_distance(self, a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _manhattan_distance(self, a, b):
        return np.sum(np.abs(a - b))

    def _minkowski_distance(self, a, b, p):
        return np.sum(np.abs(a - b) ** p) ** (1 / p)

    def _jaccard_distance(self, a, b):
        # Calculate Jaccard distance for binary vectors
        intersection = np.sum(np.minimum(a, b))
        union = np.sum(np.maximum(a, b))
        return 1 - (intersection / union) if union != 0 else 0

    def _calculate_distance(self, x, c):
        if self.distance_metric == DistanceMetric.EUCLIDEAN:
            return self._euclidean_distance(x, c)
        elif self.distance_metric == DistanceMetric.COSINE:
            return self._cosine_distance(x, c)
        elif self.distance_metric == DistanceMetric.MANHATTAN:
            return self._manhattan_distance(x, c)
        elif self.distance_metric == DistanceMetric.MINKOWSKI:
            return self._minkowski_distance(x, c, self.p)
        elif self.distance_metric == DistanceMetric.JACCARD:
            return self._jaccard_distance(x, c)
        else:
            raise ValueError("Unsupported distance metric: " + str(self.distance_metric))

    def _initialize_centroids(self, data):
        n_samples = data.shape[0]
        centroids = []

        # Randomly choose the first centroid
        first_centroid_idx = np.random.choice(n_samples)
        centroids.append(data[first_centroid_idx])

        for _ in range(1, self.n_clusters):
            # Calculate the distance of each point to the nearest centroid
            distances = np.array([min(self._calculate_distance(x, c) for c in centroids) for x in data])

            # Choose a new centroid with a probability proportional to the distance
            probabilities = distances / distances.sum()
            new_centroid_idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(data[new_centroid_idx])

        return np.array(centroids)

    def fit(self, data):
        # Normalize data if using cosine similarity
        if self.distance_metric == DistanceMetric.COSINE:
            data = data / np.linalg.norm(data, axis=1, keepdims=True)

        # Initialize centroids
        self.centroids = self._initialize_centroids(data)

        for iteration in range(self.max_iter):
            # Assign clusters based on distance
            distances = np.array([[self._calculate_distance(x, c) for c in self.centroids] for x in data])
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # Normalize new centroids if using cosine similarity
            if self.distance_metric == DistanceMetric.COSINE:
                new_centroids = new_centroids / np.linalg.norm(new_centroids, axis=1, keepdims=True)

            # Check for convergence
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

        self.labels_ = labels
        return labels

    def predict(self, data):
        """
        Predict the closest cluster for each data point.

        Parameters:
        - data: np.ndarray
            The input data to assign to clusters.

        Returns:
        - labels: np.ndarray
            The cluster labels for each data point.
        """
        # Normalize data if using cosine similarity
        if self.distance_metric == DistanceMetric.COSINE:
            data = data / np.linalg.norm(data, axis=1, keepdims=True)

        # Assign clusters based on distance
        distances = np.array([[self._calculate_distance(x, c) for c in self.centroids] for x in data])
        labels = np.argmin(distances, axis=1)
        return labels

    def plot_clusters(self, data):
        """
        Plot the clusters and centroids for 2D data.

        Parameters:
        - data: np.ndarray
            The input data that was used for clustering.
        """
        if data.shape[1] != 2:
            raise ValueError("Plotting is only supported for 2D data.")
        
        if self.labels_ is None:
            raise RuntimeError("The model must be fitted before plotting clusters.")
        
        plt.figure(figsize=(8, 6))

        # Plot each cluster
        for i in range(self.n_clusters):
            cluster_data = data[self.labels_ == i]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {i}")

        # Plot centroids
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
        plt.title(f'KMeans Clustering ({self.distance_metric.value} distance)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True)
        plt.show()