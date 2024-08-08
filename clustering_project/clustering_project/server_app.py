"""school-project: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from clustering_project.server.clustered_client_manager import SimpleClusteredClientManager
from clustering_project.server.clustered_fedavg import ClusteredFedAvg
from clustering_project.server.pfl_server import PFLServer
from clustering_project.task import Net, get_weights


# Initialize model parameters
ndarrays = get_weights(Net())
default_key = "A"
parameters_dict = {default_key: ndarrays_to_parameters(ndarrays)}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    def custom_on_fit_config_fn(partition_id: int):
        return {"partition_id": partition_id}
    # Define strategy
    strategy = ClusteredFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters_dict,
        on_fit_config_fn=custom_on_fit_config_fn
    )
    config = ServerConfig(num_rounds=num_rounds)
    client_manager = SimpleClusteredClientManager(seed=None, default_key=default_key)

    return ServerAppComponents(strategy=strategy, config=config, client_manager=client_manager,
                               server=PFLServer(client_manager=client_manager, strategy=strategy))


# Create ServerApp
app = ServerApp(server_fn=server_fn)
