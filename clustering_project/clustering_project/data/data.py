import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, DistributionPartitioner, ExponentialPartitioner, InnerDirichletPartitioner, LinearPartitioner, NaturalIdPartitioner, PathologicalPartitioner, ShardPartitioner, SizePartitioner, SquarePartitioner

fds = None

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
        trainloader = DataLoader(partition_train_test["train"], batch_size=32)
        testloader = DataLoader(partition_train_test["test"], batch_size=32)
        testloaders.append(testloader)
        trainloaders.append(trainloader)

    return trainloaders, testloaders