from flwr.server.strategy import Strategy

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg
from flwr.server.strategy import Strategy

from .clustered_client_manager import ClusteredClientManager
from .pfl_strategy import PFLStrategy

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


class ClusteredFedAvg(PFLStrategy):

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Dict[str, Parameters]] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace
        self.clusters = {}
        self.cluster_keys = []
        log(WARNING, "initial min evaluate clients: {}".format(self.min_evaluate_clients))


    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"ClusteredFedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients_in_cluster(self, num_available_clients: int, cluster_key: str) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)  # TODO enable individual min_fit, fraction_fit by cluster?
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients_in_cluster(self, num_available_clients: int, cluster_key: str) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)  # TODO enable individual min_evaluate, fraction_evaluate by cluster?
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(self, client_manager: ClusteredClientManager) -> Optional[Dict[str, Parameters]]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def configure_fit(self, server_round, parameters_dict, client_manager: ClusteredClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # fit_ins = FitIns(parameters, config)
        if server_round <= 1:
            client_manager.wait_for(num_clients=10, timeout=8400)  # TODO read config variable

        config_fits = []
        cluster_partition_ids = client_manager.get_cluster_partition_ids()
        for cluster_key in client_manager.get_cluster_keys():
            # Sample clients
            max_clients_in_cluster = client_manager.max_num_in_cluster(cluster_key=cluster_key)
            if max_clients_in_cluster > 1:
                sample_size, min_num_clients = self.num_fit_clients_in_cluster(
                    client_manager.num_available_in_cluster(cluster_key), cluster_key=cluster_key
                )
            else:
                sample_size, min_num_clients = 1, 1
            log(WARNING, "sample size {}, min num {}, available in cluster {}: {}".format(sample_size, min_num_clients, cluster_key, client_manager.num_available_in_cluster(cluster_key)))
            clients_cluster = client_manager.sample_cluster(
                num_clients=sample_size, min_num_clients=min_num_clients, cluster_key=cluster_key
            )
            for idx, partition_id in enumerate(cluster_partition_ids[cluster_key]):
                config = {}
                if self.on_fit_config_fn is not None:
                    # Custom fit config function provided
                    # log(WARNING, "Custom fit config")
                    config = self.on_fit_config_fn(partition_id=partition_id)
                else:
                    log(WARNING, "Custom fit config is None")
                fit_ins = FitIns(parameters_dict[cluster_key], config)
                config_fits += [(clients_cluster[idx], fit_ins)]

        # Return client/config pairs
        return config_fits

    def _sort_fits_by_cluster(self, client_manager: ClusteredClientManager, fits: List[Tuple[ClientProxy, Union[FitRes, EvaluateRes]]],
                              ) -> Dict[str, List[Union[Tuple[ClientProxy, FitRes], BaseException]]]:
        # TODO handle failures containing exceptions
        # TODO identify clients by cid instead of client proxy
        clients_by_cluster = client_manager.all_by_cluster()
        fits_by_cluster = {cluster_key: [] for cluster_key in clients_by_cluster.keys()}
        for (client, fit_res) in fits:
            for cluster_key, cluster_clients in clients_by_cluster.items():
                if client in [proxy for proxy in cluster_clients.values()]:
                    fits_by_cluster[cluster_key] += [(client, fit_res)]
        return fits_by_cluster

    def aggregate_fit(self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        client_manager: ClusteredClientManager,
    ) -> Tuple[Optional[Dict[str, Parameters]], Dict[str, Dict[str, Scalar]]]:
        """Aggregate fit results using weighted average."""
        # TODO handle these by cluster
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        results_by_cluster = self._sort_fits_by_cluster(client_manager=client_manager, fits=results)
        # failures_by_cluster = self._sort_fits_by_cluster(client_manager=client_manager, fits=failures)

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays_dict = {}
            for cluster_key, cluster_results in results_by_cluster.items():
                aggregated_ndarrays_dict[cluster_key] = aggregate_inplace(cluster_results)
        else:
            # Convert results
            aggregated_ndarrays_dict = {}
            for cluster_key, cluster_results in results_by_cluster.items():
                weights_results = [
                    (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                    for _, fit_res in cluster_results
                ]
                aggregated_ndarrays_dict[cluster_key] = aggregate(weights_results)

        parameters_aggregated_dict = {cluster_key: ndarrays_to_parameters(aggregated_ndarrays) for cluster_key, aggregated_ndarrays in aggregated_ndarrays_dict.items()}

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated_dict = {}
        if self.fit_metrics_aggregation_fn:
            for cluster_key, cluster_results in results_by_cluster.items():
                fit_metrics = [(res.num_examples, res.metrics) for _, res in cluster_results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
                metrics_aggregated_dict[cluster_key] = metrics_aggregated
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated_dict, metrics_aggregated_dict

    def configure_evaluate(self, server_round: int, parameters_dict: Dict[str, Parameters], client_manager: ClusteredClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)

        config_evals = []
        for cluster_key in client_manager.get_cluster_keys():
            evaluate_ins = EvaluateIns(parameters_dict[cluster_key], config)
            max_clients_in_cluster = client_manager.max_num_in_cluster(cluster_key=cluster_key)
            # Sample clients
            if max_clients_in_cluster > 1:
                sample_size, min_num_clients = self.num_fit_clients_in_cluster(
                    client_manager.num_available_in_cluster(cluster_key), cluster_key=cluster_key
                )
            else:
                sample_size, min_num_clients = 1, 1
            # sample_size, min_num_clients = self.num_evaluation_clients_in_cluster(
            #    client_manager.num_available_in_cluster(cluster_key), cluster_key=cluster_key
            # )
            log(WARNING, "eval sample size {}, min num {}, available in cluster {}: {}".format(sample_size, min_num_clients, cluster_key, client_manager.num_available_in_cluster(cluster_key)))
            clients = client_manager.sample_cluster(
                num_clients=sample_size, min_num_clients=min_num_clients, cluster_key=cluster_key
            )
            config_evals += [(client, evaluate_ins) for client in clients]

        # Return client/config pairs
        return config_evals

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]], client_manager: ClusteredClientManager
    ) -> Tuple[Optional[Dict[str, float]], Dict[str, Dict[str, Scalar]]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        results_by_cluster = self._sort_fits_by_cluster(client_manager=client_manager, fits=results)

        # Aggregate loss
        loss_aggregated_by_cluster = {}
        for cluster_key, cluster_results in results_by_cluster.items():
            loss_aggregated = weighted_loss_avg(
                [
                    (evaluate_res.num_examples, evaluate_res.loss)
                    for _, evaluate_res in cluster_results
                ]
            )
            loss_aggregated_by_cluster[cluster_key] = loss_aggregated

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated_by_cluster = {cluster_key: {} for cluster_key in client_manager.get_cluster_keys()}
        if self.evaluate_metrics_aggregation_fn:
            metrics_aggregated = {}
            for cluster_key, cluster_results in results_by_cluster.items():
                eval_metrics = [(res.num_examples, res.metrics) for _, res in cluster_results]
                metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
                metrics_aggregated_by_cluster[cluster_key] = metrics_aggregated
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated_by_cluster, metrics_aggregated_by_cluster

    def evaluate(
            self, server_round: int, parameters_dict: Dict[str, Parameters]
    ) -> Optional[Tuple[Dict[str, float], Dict[str, Dict[str, Scalar]]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays_dict = {cluster_key: parameters_to_ndarrays(parameters)
                                    for cluster_key, parameters in parameters_dict.items()}
        loss_dict = {}
        metrics_dict = {}
        for key, parameters_ndarrays in parameters_ndarrays_dict.items():
            eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
            if eval_res is None:
                return None
            loss, metrics = eval_res
            loss_dict[key] = loss
            metrics_dict[key] = metrics
        return loss_dict, metrics_dict