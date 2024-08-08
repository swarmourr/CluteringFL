"""benchmark: A Flower / PyTorch app."""

from typing import List, Tuple
from flwr.common import Metrics, EvaluateRes
import pandas as pd

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from benchmark.task import Net, get_weights

# Initialize model parameters
ndarrays = get_weights(Net())
parameters = ndarrays_to_parameters(ndarrays)

# Function to save client accuracies to CSV
def save_accuracies_to_csv(rnd: int, results: List[Tuple[int, EvaluateRes]], filename: str = "client_accuracies.csv"):
    # Extract partition IDs and their accuracies
    partition_ids = [evaluate_res.metrics["partition_id"] for _, evaluate_res in results]
    accuracies = [evaluate_res.metrics["accuracy"] for _, evaluate_res in results]
    losses = [evaluate_res.loss for _, evaluate_res in results]
    
    # Create a DataFrame and append to CSV
    df = pd.DataFrame({"Round": [rnd]*len(partition_ids), "Partition ID": partition_ids, "Accuracy": accuracies, "Loss": losses})
    
    # If the file exists, append without header, otherwise include header
    if rnd == 1:
        df.to_csv(filename, index=False, mode='w')  # Write mode for the first round
    else:
        df.to_csv(filename, index=False, header=False, mode='a')  # Append mode for subsequent rounds

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by the number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

class CustomFedAvg(FedAvg):
    def aggregate_evaluate(self, rnd: int, results: List[Tuple[int, EvaluateRes]], failures: List[BaseException]):
        if not results:
            return None, {}
        
        # Save accuracies to CSV
        save_accuracies_to_csv(rnd, results)

        # Call parent method to perform standard aggregation
        return super().aggregate_evaluate(rnd, results, failures)

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Define strategy
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)