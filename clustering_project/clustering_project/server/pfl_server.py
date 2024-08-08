# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower server."""


import concurrent.futures
import copy
import io
import timeit
from logging import INFO, WARN
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
)
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy

from flwr.server.server_config import ServerConfig
from flwr.server import Server

from school_project.server.clustered_fedavg import ClusteredFedAvg
from school_project.server.pfl_strategy import PFLStrategy

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]


class PFLServer(Server):  # (Server):
    """Flower server for personalised FL."""

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        self._client_manager: ClientManager = client_manager
        default_key = self._client_manager.get_default_key()
        self.parameters_dict: Dict[str, Parameters] = {default_key: Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )}  # can hold an arbitrary number of personalised models; default is one (equivalent to non-PFL)
        log(INFO, "strategy is None %s".format(strategy is None))
        self.strategy: PFLStrategy = strategy if strategy is not None else ClusteredFedAvg()
        self.max_workers: Optional[int] = None

    def set_max_workers(self, max_workers: Optional[int]) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers

    def set_strategy(self, strategy: PFLStrategy) -> None:
        """Replace server strategy."""
        self.strategy = strategy

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> Tuple[History, float]:
        """Run personalised federated learning for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "[INIT]")
        self.parameters_dict = self._get_initial_parameters(server_round=0, timeout=timeout)
        log(INFO, "Evaluating initial global parameters")
        loss_mean = 0.
        metrics_mean = {}
        results_valid = True
        res_dict = self.strategy.evaluate(0, parameters_dict=self.parameters_dict)
        if res_dict is not None:
            loss_dict, metrics_dict = res_dict
            for key in loss_dict.keys():
                log(
                        INFO,
                        "initial parameters (ID, loss, other metrics): %s, %s, %s",
                        key,
                        loss_dict[key],
                        metrics_dict[key],
                )
                loss_mean += loss_dict[key]/len(self.parameters_dict)
                for metric_key, metric_value in metrics_dict[key].items():
                    if metric_key in metrics_mean:
                        metrics_mean[metric_key] = metrics_mean[metric_key]+metric_value/len(self.parameters_dict)
                    else:
                        metrics_mean[metric_key] = metric_value/len(self.parameters_dict)
        else:
            results_valid = False  # flag for logging
        if results_valid:
            history.add_loss_centralized(server_round=0, loss=loss_mean)
            history.add_metrics_centralized(server_round=0, metrics=metrics_mean)

        # Run federated learning for num_rounds
        start_time = timeit.default_timer()


        # split_key = "A"
        # self._client_manager.split_cluster(split_key)
        #TODO do this only if clustering based on data
        log(INFO, "[CLUSTERING before round 1]")
        self._client_manager.compute_clusters_data()
        current_cluster_keys = self._client_manager.get_cluster_keys()
        split_key = "A"  # TODO this only works if we cluster only once -> need to map old cluster models to new ones (or recompute)
        for cluster_key in current_cluster_keys:
            if cluster_key not in self.parameters_dict.keys():
                self.parameters_dict[cluster_key] = copy.deepcopy(self.parameters_dict[split_key])
        del self.parameters_dict[split_key]

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s]", current_round)
            # Train personalised models and replace previous global model(s)
            res_fits = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fits is not None:
                parameters_prime_dict, fit_metrics_dict, _ = res_fits  # fit_metrics_aggregated
                if parameters_prime_dict:
                    self.parameters_dict = parameters_prime_dict
                fit_metrics = {}
                for key, metrics in fit_metrics_dict.items():
                    fit_metrics[key] = fit_metrics[key] + metrics[key]/len(fit_metrics_dict) \
                        if key in fit_metrics else metrics[key]/len(fit_metrics_dict)
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            loss_cen_mean = 0.
            metrics_cen_mean = {}
            results_cen_valid = True

            res_cen_dicts = self.strategy.evaluate(current_round, parameters_dict=self.parameters_dict)
            if res_cen_dicts is not None:
                loss_cen_dict, metrics_cen_dict = res_cen_dicts
                for key in loss_cen_dict.keys():
                    loss_cen, metrics_cen = loss_cen_dict[key], metrics_cen_dict[key]
                    log(
                        INFO,
                        "fit progress: (ID %s, %s, %s, %s, %s)",
                        key,
                        current_round,
                        loss_cen,
                        metrics_cen,
                        timeit.default_timer() - start_time,
                    )
                    loss_cen_mean += loss_cen / len(loss_cen_dict)
                    for metric_key, metric_value in metrics_cen.items():
                        if metric_key in metrics_cen_mean:
                            metrics_cen_mean[metric_key] = metrics_cen_mean[metric_key] + metric_value / len(metrics_cen_dict)
                        else:
                            metrics_cen_mean[metric_key] = metric_value / len(metrics_cen_dict)
            else:
                results_cen_valid = False  # flag for logging TODO
            if results_cen_valid:
                history.add_loss_centralized(server_round=current_round, loss=loss_cen_mean)
                history.add_metrics_centralized(
                       server_round=current_round, metrics=metrics_cen_mean
                )

            # Evaluate model on a sample of available clients
            loss_fed_mean = 0.
            metrics_fed_mean = {}
            results_fed_valid = True
            res_fed_dicts = self.evaluate_round(server_round=current_round, timeout=timeout)

            if res_fed_dicts is not None:
                loss_fed_dict, evaluate_metrics_fed_dict, _ = res_fed_dicts
                for model_key in loss_fed_dict.keys():
                    loss_fed = loss_fed_dict[model_key]
                    evaluate_metrics_fed = evaluate_metrics_fed_dict[model_key]
                    if loss_fed is not None:
                        loss_fed_mean += loss_fed / len(self.parameters_dict)
                        for metric_key, metric_value in evaluate_metrics_fed.items():
                            if metric_key in metrics_cen_mean:
                                metrics_fed_mean[metric_key] = metrics_fed_mean[metric_key] + metric_value / len(
                                    self.parameters_dict)
                            else:
                                metrics_fed_mean[metric_key] = metric_value / len(self.parameters_dict)
                    else:
                        results_fed_valid = False  # flag for logging

            if results_fed_valid:
                history.add_loss_distributed(
                     server_round=current_round, loss=loss_fed_mean
                )
                history.add_metrics_distributed(
                     server_round=current_round, metrics=metrics_fed_mean
                )
        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        return history, elapsed

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Dict[str, float]], Dict[str, Dict[str, Scalar]], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters_dict=self.parameters_dict,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "configure_evaluate: no clients selected, skipping evaluation")
            return None
        log(
            INFO,
            "configure_evaluate: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            INFO,
            "aggregate_evaluate: received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[Dict[str, float]],
            Dict[str, Dict[str, Scalar]],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures, client_manager=self._client_manager)

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Dict[str, Parameters]], Dict[str, Dict[str, Scalar]], FitResultsAndFailures]
    ]:
        """Perform a single round of personalised federated learning."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters_dict=self.parameters_dict,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "configure_fit: no clients selected, cancel")
            return None
        log(
            INFO,
            "configure_fit: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            INFO,
            "aggregate_fit: received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Dict[str, Parameters]],
            Dict[str, Dict[str, Scalar]],
        ] = self.strategy.aggregate_fit(server_round, results, failures, client_manager=self._client_manager)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)

    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        clients = [all_clients[k] for k in all_clients.keys()]
        instruction = ReconnectIns(seconds=None)
        client_instructions = [(client_proxy, instruction) for client_proxy in clients]
        _ = reconnect_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

    def _get_initial_parameters(
        self, server_round: int, timeout: Optional[float]
    ) -> Dict[str, Parameters]:
        """Get initial parameters from one of the available clients."""
        # Server-side parameter initialization
        parameters: Optional[Dict[str, Parameters]] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial global parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        parameters = {}
        log(INFO, "Requesting initial parameters from all clients")
        for cid, client in self._client_manager.all().items():
            # random_client = self._client_manager.sample(1)[0]
            ins = GetParametersIns(config={})
            get_parameters_res = client.get_parameters(
                ins=ins, timeout=timeout, group_id=server_round
            )
            parameters[cid] = get_parameters_res.parameters
            if get_parameters_res.status.code == Code.OK:
                log(INFO, "Received initial parameters from client {}".format(cid))
            else:
                log(
                    WARN,
                    "Failed to receive initial parameters from client {}.".format(cid) +
                    " Empty initial parameters will be used.",
                )
        return parameters


def reconnect_clients(
    client_instructions: List[Tuple[ClientProxy, ReconnectIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(reconnect_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, DisconnectRes]] = []
    failures: List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy,
    reconnect: ReconnectIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, DisconnectRes]:
    """Instruct client to disconnect and (optionally) reconnect later."""
    disconnect = client.reconnect(
        reconnect,
        timeout=timeout,
        group_id=None,
    )
    return client, disconnect


def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
    group_id: int,
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout, group_id)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    return results, failures


def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float], group_id: int
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout, group_id=group_id)
    return client, fit_res


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, FitRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
    group_id: int,
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client, client_proxy, ins, timeout, group_id)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_evaluate(
            future=future, results=results, failures=failures
        )
    return results, failures


def evaluate_client(
    client: ClientProxy,
    ins: EvaluateIns,
    timeout: Optional[float],
    group_id: int,
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins, timeout=timeout, group_id=group_id)
    return client, evaluate_res


def _handle_finished_future_after_evaluate(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, EvaluateRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def init_defaults(
    server: Optional[Server],
    config: Optional[ServerConfig],
    strategy: Optional[Strategy],
    client_manager: Optional[ClientManager],
) -> Tuple[Server, ServerConfig]:
    """Create server instance if none was given."""
    if server is None:
        if client_manager is None:
            client_manager = SimpleClientManager()
        if strategy is None:
            strategy = FedAvg()
        server = Server(client_manager=client_manager, strategy=strategy)
    elif strategy is not None:
        log(WARN, "Both server and strategy were provided, ignoring strategy")

    # Set default config values
    if config is None:
        config = ServerConfig()

    return server, config


def run_fl(
    server: Server,
    config: ServerConfig,
) -> History:
    """Train a model on the given server and return the History object."""
    hist, elapsed_time = server.fit(
        num_rounds=config.num_rounds, timeout=config.round_timeout
    )

    log(INFO, "")
    log(INFO, "[SUMMARY]")
    log(INFO, "Run finished %s round(s) in %.2fs", config.num_rounds, elapsed_time)
    for line in io.StringIO(str(hist)):
        log(INFO, "\t%s", line.strip("\n"))
    log(INFO, "")

    # Graceful shutdown
    server.disconnect_all_clients(timeout=config.round_timeout)

    return hist
