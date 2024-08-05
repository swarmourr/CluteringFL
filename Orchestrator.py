import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from Clustering.FLKmeans import KMeans
from Clustering.Distances import DistanceMetric

class Orchestrator:
    def __init__(self, distance_metric=DistanceMetric.EUCLIDEAN, max_clusters=10):
        """
        Initializes the Orchestrator object.

        Parameters:
        - distance_metric (DistanceMetric): The distance metric to be used for clustering. Default is DistanceMetric.EUCLIDEAN.
        - max_clusters (int): The maximum number of clusters to evaluate. Default is 10.
        """
        self.distance_metric = distance_metric
        self.max_clusters = max_clusters
        self._optimal_clusters = None
        self._centroids = None
        self._labels = None
        self.original_labels = None  # Store original labels for data points

    @property
    def optimal_clusters(self):
        """
        Gets the optimal number of clusters.
        """
        return self._optimal_clusters

    @optimal_clusters.setter
    def optimal_clusters(self, value):
        """
        Sets the optimal number of clusters.

        Parameters:
        - value (int): The new value for the optimal number of clusters.
        """
        if not isinstance(value, int) or value < 2:
            raise ValueError("Optimal clusters must be an integer greater than or equal to 2.")
        self._optimal_clusters = value

    def cluster_data(self, data: pd.DataFrame, n_clusters: int):
        """
        This method is responsible for clustering the input data.

        Parameters:
        - data (pd.DataFrame): The input data to be clustered.
        - n_clusters (int): The number of clusters to form.

        Returns:
        - labels (np.ndarray): The cluster labels for each data point.
        """
        # Initialize KMeans with the specified number of clusters
        kmeans = KMeans(n_clusters=n_clusters, distance_metric=self.distance_metric)
        
        # Fit the KMeans algorithm on the data
        labels = kmeans.fit(data)
        
        # Set centroids for access later
        self._centroids = kmeans.centroids

        # Return the cluster labels
        return labels

    def find_optimal_clusters(self, data: pd.DataFrame, original_labels=None):
        """
        This method is responsible for finding the optimal number of clusters in the data.

        Parameters:
        - data (pd.DataFrame): The input data for which to find the optimal number of clusters.
        - original_labels (list or pd.Series): Original labels for each data point (e.g., client IDs).
        """
        max_score = -1
        optimal_clusters = 2
        self.original_labels = original_labels  # Store original labels

        for i in range(2, self.max_clusters + 1):
            # Cluster data and get labels
            labels = self.cluster_data(data, n_clusters=i)
            
            # Compute the silhouette score for the current number of clusters
            silhouette_avg = silhouette_score(data, labels)
            
            # Print the silhouette score
            print(f"Silhouette Score for {i} clusters: {silhouette_avg:.3f}")

            # Determine if this is the best score
            if silhouette_avg > max_score:
                max_score = silhouette_avg
                optimal_clusters = i
                self._labels = labels
        
        self.optimal_clusters = optimal_clusters
        print(f"Optimal number of clusters based on silhouette score: {self.optimal_clusters}")

    def get_cluster_info(self):
        """
        Returns the labels and centroids of the optimal clustering.

        Returns:
        - labels (np.ndarray): Cluster labels of the data.
        - centroids (np.ndarray): Centroids of the clusters.
        """
        if self._labels is None or self._centroids is None:
            raise ValueError("Clusters have not been computed. Call 'find_optimal_clusters' first.")
        return self._labels, self._centroids

    def get_cluster_mapping(self):
        """
        Creates a mapping of cluster IDs to the original labels of data points assigned to each cluster.

        Returns:
        - cluster_mapping (dict): A dictionary mapping each cluster ID to a list of original labels.
        """
        if self._labels is None:
            raise ValueError("Clusters have not been computed. Call 'find_optimal_clusters' first.")
        
        cluster_mapping = {}
        for idx, label in enumerate(self._labels):
            if label not in cluster_mapping:
                cluster_mapping[label] = []
            original_label = self.original_labels[idx] if self.original_labels is not None else idx
            cluster_mapping[label].append(original_label)
        
        return cluster_mapping


# Generate some sample data
data, _ = make_blobs(n_samples=300, centers=5, cluster_std=1.0, random_state=42)

# Convert to DataFrame for processing
data = pd.DataFrame(data)

# Add a column of original labels (e.g., client IDs)
original_labels = [f"Client_{i+1}" for i in range(data.shape[0])]
data['ClientID'] = original_labels


print(data)
# Optionally standardize the data (excluding ClientID column)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop(columns=['ClientID']))

# Initialize the Orchestrator
orchestrator = Orchestrator(max_clusters=10)

# Find the optimal clusters with original labels
orchestrator.find_optimal_clusters(data_scaled, original_labels=original_labels)

# Get cluster information
labels, centroids = orchestrator.get_cluster_info()
print("Labels of the optimal clustering:", labels)
print("Centroids of the optimal clustering:\n", centroids)

# Get the mapping between cluster IDs and original labels
cluster_mapping = orchestrator.get_cluster_mapping()
print("Cluster to Original Label Mapping:", cluster_mapping)
