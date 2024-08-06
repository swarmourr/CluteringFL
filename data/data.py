import os
import tensorflow as tf
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, DistributionPartitioner, ExponentialPartitioner, InnerDirichletPartitioner, LinearPartitioner, NaturalIdPartitioner, PathologicalPartitioner, ShardPartitioner, SizePartitioner, SquarePartitioner
from tqdm import tqdm
import pandas as pd 

fds= None 

def load_data(data_name, n_partitioner, chosen_partitioner):
    # Download and partition dataset
    # Only initialize `FederatedDataset` once
    # Possible partitioners: https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html

    # Make directories for storing datasets
    if not os.path.exists("data"):
        os.mkdir("data")

    if not os.path.exists(f"data/{data_name}"):
        os.mkdir(f"data/{data_name}")

    if not os.path.exists(f"data/{data_name}/{n_partitioner}_partitions"):
        os.mkdir(f"data/{data_name}/{n_partitioner}_partitions")

    # Load data from flower
    global fds
    if fds is None:
        fds = FederatedDataset(
            dataset=data_name,
            partitioners={"train": chosen_partitioner},
            seed=42
        )
    
    # Save each partition to a csv file
    for partition in tqdm(range(n_partitioner)):
        part = fds.load_partition(partition)
        part.set_format("numpy")
        df = pd.DataFrame(part)
        df.to_csv(f"data/{data_name}/{n_partitioner}_partitions/{partition}.csv", index=False)


load_data("mnist", 10, IidPartitioner(10))
