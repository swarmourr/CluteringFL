"""school-project: A Flower / PyTorch app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from clustering_project.task import (
    Net,
    DEVICE,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, partition_id, num_partitions):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.partition_id = partition_id
        self.num_partitions = num_partitions

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        self.partition_id = config['partition_id']

        self.trainloader, self.valloader = load_data(self.partition_id, self.num_partitions)
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            DEVICE,
        )
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy, 'partition_id':self.partition_id}


def client_fn(context: Context):
    # Load model and data
    net = Net().to(DEVICE)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs, partition_id=partition_id, num_partitions=num_partitions).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
