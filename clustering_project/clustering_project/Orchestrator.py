import json
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from clustering_project.Clustering.FLKmeans import KMeans
from clustering_project.Clustering.Distances import DistanceMetric
from clustering_project.Statistics.stats import DataFrameStatistics
from clustering_project.Statistics.model import GradientExtractor, SimpleModel

import glob

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, DistributionPartitioner, ExponentialPartitioner, InnerDirichletPartitioner, LinearPartitioner, NaturalIdPartitioner, PathologicalPartitioner, ShardPartitioner, SizePartitioner, SquarePartitioner


fds = None


def flatten_image_dataframe(df):
    def convert_string_to_array(s):
        # Remove extra brackets and split by spaces
        numbers = s.strip('[]').replace('[', '').replace(']', '').split()
        # Convert to integers
        return np.array([int(num) for num in numbers if num])

    # Apply the conversion to the 'image' column
    df['image'] = df['image'].apply(convert_string_to_array)

    # Convert the series of arrays into a 2D numpy array
    image_arrays = np.stack(df['image'].values)

    # Create a new DataFrame with each pixel as a feature
    pixel_df = pd.DataFrame(image_arrays, columns=[f'pixel_{i}' for i in range(image_arrays.shape[1])])

    # Add the label column
    pixel_df['label'] = df['label']

    return pixel_df


def load_data(num_partitions: int, partitioner):
    """Load partition MNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        fds = FederatedDataset(
            dataset="mnist",
            partitioners={"train": partitioner,
                          },
            trust_remote_code=True
        )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    trainloaders = []
    testloaders = []

    for partition_id in range(num_partitions):
        partition = fds.load_partition(partition_id)
        partition = partition.rename_column("image", "img")
        # Divide data on each node: 80% train, 20% test
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        pytorch_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        # pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # For CIFAR10

        partition_train_test = partition_train_test.with_transform(apply_transforms)
        trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
        testloader = DataLoader(partition_train_test["test"], batch_size=32)
        testloaders.append(testloader)
        trainloaders.append(trainloader)

    return trainloaders, testloaders


class Client:
    def __init__(self, client_id, data_arrays=None, column_names=None):
        """
        Initializes the Client object.

        Parameters:
        - client_id: int or str, unique identifier for the client
        - data_arrays: list of np.ndarray, list of NumPy arrays representing client data
        """
        self.client_id = client_id
        # Initialize data_arrays as an empty list if not provided
        self.data_arrays = data_arrays if data_arrays is not None else []
        # Initialize data_arrays as an empty list if not provided
        self.data_arrays = data_arrays if data_arrays is not None else []
        self.column_names = column_names if column_names is not None else []

    def add_array(self, new_array, column_names=None):
        """
        Adds a new NumPy array to the data_arrays list.

        Parameters:
        - new_array: np.ndarray, the NumPy array to be added
        """
        if not isinstance(new_array, np.ndarray):
            raise ValueError("The new_array must be a NumPy ndarray.")

        self.data_arrays.append(new_array)

    def _generate_default_column_names(self, num_columns):
        """
        Generates default column names.

        Parameters:
        - num_columns: int, number of columns needed

        Returns:
        - list of str, default column names
        """
        return [f"col_{i + 1}" for i in range(num_columns)]

    def to_dataframe(self):
        """
        Transforms the list of NumPy arrays into a single pandas DataFrame.

        Returns:
        - df: pd.DataFrame, concatenated DataFrame of all NumPy arrays
        """
        # Check if the list is empty
        if not self.data_arrays:
            return pd.DataFrame()

        # Convert each NumPy array to a DataFrame
        dfs = [pd.DataFrame(array) for array in self.data_arrays]

        # Concatenate all DataFrames along the columns
        df = pd.concat(dfs, axis=1)

        # Set column names if provided, else use default names
        if self.column_names:
            df.columns = self.column_names[:df.shape[1]]  # Slice column_names to match the number of columns
        else:
            df.columns = self._generate_default_column_names(df.shape[1])

        return df

    def __repr__(self):
        return f"Client(client_id={self.client_id}, num_arrays={len(self.data_arrays)})"


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

    def find_optimal_clusters(self, data: pd.DataFrame):
        """
        This method is responsible for finding the optimal number of clusters in the data.

        Parameters:
        - data (pd.DataFrame): The input data for which to find the optimal number of clusters.
        """
        max_score = -1
        optimal_clusters = 2

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

    def get_cluster_mapping(self, original_labels):
        """
        Creates a mapping of cluster IDs to the original labels of data points assigned to each cluster.

        Parameters:
        - original_labels (list): The list of original labels (e.g., client names or IDs) for each data point.

        Returns:
        - cluster_mapping (dict): A dictionary mapping each cluster ID to a list of original labels.
        """
        if self._labels is None:
            raise ValueError("Clusters have not been computed. Call 'find_optimal_clusters' first.")

        cluster_mapping = {}
        for idx, label in enumerate(self._labels):
            # Convert the cluster ID to a standard Python integer
            cluster_id = int(label)
            if cluster_id not in cluster_mapping:
                cluster_mapping[cluster_id] = []
            cluster_mapping[int(cluster_id)].append(int(original_labels[idx]))

        return cluster_mapping

    def export_cluster_mapping_to_json(self, cluster_mapping, filename="cluster_mapping.json"):
        """
        Exports the cluster mapping to a JSON file.

        Parameters:
        - cluster_mapping (dict): The cluster mapping dictionary.
        - filename (str): The name of the file to save the JSON output. Default is 'cluster_mapping.json'.
        """
        # Convert keys to strings to ensure compatibility with JSON
        json_compatible_mapping = {str(k): v for k, v in cluster_mapping.items()}

        with open(filename, 'w') as f:
            json.dump(json_compatible_mapping, f, indent=4)
        print(f"Cluster mapping exported to {filename}")

    def main_data(self, trains, distance_metric=DistanceMetric.EUCLIDEAN):
        """
        Process and cluster data from the given data loaders.

        Parameters:
        - trains: List of DataLoader objects, each representing client training data.
        - distance_metric (DistanceMetric): The distance metric to use for clustering.

        Returns:
        - cluster_mapping: A dictionary mapping each cluster ID to a list of client IDs.
        """
        trains, testloaders = load_data(10,
                                              DirichletPartitioner(num_partitions=10, partition_by="label", alpha=0.5,
                                                                   min_partition_size=10, self_balancing=True))

        Client_list = list()
        client_idx = list()
        stats = pd.DataFrame()
        for idx, train in enumerate(trains):
            print(idx)
            client = Client(idx)
            Client_list.append(client)
            for batch in train:
                for batch_one in batch['img']:
                    client.add_array(batch_one.numpy().flatten())
        for client in Client_list:
            print(client)
            data = client.to_dataframe()
            df_stats = DataFrameStatistics(data)
            all_stats = DataFrameStatistics(data).all_statistics()
            single_row_df = df_stats.create_feature_stat_df(all_stats)
            client_idx.append(client.client_id)
            stats = pd.concat([stats, single_row_df])
        stats = (stats.notnull()).astype('int')

        # Optionally standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(stats)

        # Initialize the Orchestrator
        print("clustering")
        orchestrator = Orchestrator(max_clusters=len(client_idx) - 1, distance_metric=distance_metric)
        # Find the optimal clusters
        orchestrator.find_optimal_clusters(data_scaled)
        # Get cluster information
        labels, centroids = orchestrator.get_cluster_info()
        print("Labels of the optimal clustering:", labels)
        print("Centroids of the optimal clustering:\n", centroids)

        # Get the mapping between cluster IDs and client IDs
        cluster_mapping = orchestrator.get_cluster_mapping(client_idx)
        print("Cluster to Client Mapping:", cluster_mapping)

        # Export the mapping to a JSON file
        orchestrator.export_cluster_mapping_to_json(cluster_mapping, filename="data_cluster_mapping.json")
        return cluster_mapping

    def main_gradients(self, distance_metric=DistanceMetric.EUCLIDEAN):
        """
        Generate and cluster gradients from models.

        Parameters:
        - distance_metric (DistanceMetric): The distance metric to use for clustering.

        Returns:
        - cluster_mapping: A dictionary mapping each cluster ID to a list of model names.
        """
        # Define the model architecture parameters
        model_params = {
            'input_size': 10,
            'hidden_size': 5,
            'output_size': 2
        }

        # Number of models to create
        num_models = 5

        # Create GradientExtractor instance
        extractor = GradientExtractor(SimpleModel, model_params, num_models)

        # Extract gradients from all models
        gradients_df = extractor.extract_gradients()


        print("Combined Gradients DataFrame:")
        print(gradients_df)

        # Optional: Compute some statistics on the gradients
        print("\nGradient Statistics:")
        for name in extractor.model_names:
            model_gradients = gradients_df[gradients_df['model_name'] == name].drop('model_name', axis=1)
            print(f"\n{name}:")
            print(f"Mean gradient: {model_gradients.mean().mean():.6f}")
            print(f"Max gradient: {model_gradients.max().max():.6f}")
            print(f"Min gradient: {model_gradients.min().min():.6f}")

        # Initialize the Orchestrator
        print("clustering")
        model_names = gradients_df['model_name']
        gradients_df = gradients_df.drop('model_name', axis=1)

        orchestrator = Orchestrator(max_clusters=len(model_names)-1, distance_metric=distance_metric)
        # Find the optimal clusters
        orchestrator.find_optimal_clusters(gradients_df.values)
        # Get cluster information
        labels, centroids = orchestrator.get_cluster_info()
        print("Labels of the optimal clustering:", labels)
        print("Centroids of the optimal clustering:\n", centroids)

        # Get the mapping between cluster IDs and client IDs
        cluster_mapping = orchestrator.get_cluster_mapping(model_names)
        print("Cluster to Client Mapping:", cluster_mapping)

        # Export the mapping to a JSON file
        orchestrator.export_cluster_mapping_to_json(cluster_mapping, filename="cluster_mapping.json")

        return cluster_mapping



"""
# Step 1: Load the data
# Define the number of partitions (clients) and partitioning strategy
num_partitions = 5  # For example, 5 clients
partitioner = "iid"  # or other partitioning strategies like 'non-iid'

# Load the data using the load_data function
trainloaders, testloaders = load_data(10, DirichletPartitioner(num_partitions=10, partition_by="label",alpha=0.5, min_partition_size=10,self_balancing=True))


# Step 2: Initialize the Orchestrator
orchestrator = Orchestrator(distance_metric=DistanceMetric.EUCLIDEAN, max_clusters=10)

# Step 3: Call main_data to cluster the data
cluster_mapping_data = orchestrator.main_data(trains=trainloaders, distance_metric=DistanceMetric.EUCLIDEAN)

# Print the cluster mapping for the data
print("Cluster Mapping for Data:")
print(cluster_mapping_data)

# Step 4: Call main_gradients to cluster the gradients
cluster_mapping_gradients = orchestrator.main_gradients(distance_metric=DistanceMetric.EUCLIDEAN)

# Print the cluster mapping for the gradients
print("Cluster Mapping for Gradients:")
print(cluster_mapping_gradients)
"""