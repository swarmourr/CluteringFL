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
"""Flower ClientManager."""
import json
import random
import threading
from abc import ABC, abstractmethod
from logging import INFO, WARNING
from typing import Dict, List, Optional, Tuple

from flwr.common.logger import log

from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server.client_manager import ClientManager
from os.path import expanduser


class ClusteredClientManager(ClientManager):
    """Abstract base class for managing Flower clients with clustering."""

    @abstractmethod
    def num_available(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """

    @abstractmethod
    def num_available_by_cluster(self) -> Dict[str, int]:
        """Return the number of available clients per cluster.

        Returns
        -------
        num_available_by_cluster : List[int]
            The number of currently available clients, listed by cluster ID.
        """

    @abstractmethod
    def num_available_in_cluster(self, cluster_key: str) -> int:
        """Return the number of available clients per cluster.

        Parameters
        -------
        cluster_key: str ID of cluster

        Returns
        -------
        num_available_in_cluster : List[int]
            The number of currently available clients in the given cluster.
        """

    @abstractmethod
    def register_in_cluster(self, client: ClientProxy, cluster_key: str) -> bool:
        """Register Flower ClientProxy instance in a given cluster.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy
        cluster_key: str giving the ID of a cluster

        Returns
        -------
        success : bool
            Indicating if registration was successful. False if ClientProxy is
            already registered or can not be registered for any reason,
            including because the indexed cluster does not exist.
        """

    @abstractmethod
    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy

        Returns
        -------
        success : bool
            Indicating if registration was successful. False if ClientProxy is
            already registered or can not be registered for any reason.
        """

    @abstractmethod
    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance.

        This method is idempotent.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy
        """

    @abstractmethod
    def all_by_cluster(self) -> Dict[str, Dict[str, ClientProxy]]:
        """Return all available clients, with index of the current cluster."""

    @abstractmethod
    def split_cluster(self, cluster_key: str) -> bool:
        """Test method to modify clusters"""

    @abstractmethod
    def all_in_cluster(self, cluster_key: str) -> Dict[str, ClientProxy]:
        """Return all available clients for the given cluster."""

    @abstractmethod
    def all(self) -> Dict[str, ClientProxy]:
        """Return all available clients."""

    @abstractmethod
    def get_default_key(self) -> str:
        """Return the default key that labels the first cluster"""

    @abstractmethod
    def compute_clusters_data(self) -> bool:
        """load clustering mapping and convert to clusters accordingly. Disregard input if it cannot be converted."""

    @abstractmethod
    def wait_for_cluster(self, num_clients: int, timeout: int, cluster_key: str) -> bool:
        """Wait until at least `num_clients` are available in cluster cluster_key."""

    @abstractmethod
    def wait_for(self, num_clients: int, timeout: int) -> bool:
        """Wait until at least `num_clients` are available."""

    @abstractmethod
    def get_cluster_keys(self) -> List[str]:
        """Return list of keys for current clusters."""

    @abstractmethod
    def get_cluster_partition_ids(self) -> Dict[str, List[int]]:
        """Return partition IDs for clusters"""

    @abstractmethod
    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""

    @abstractmethod
    def sample_cluster(
            self,
            num_clients: int,
            cluster_key: str,
            min_num_clients: Optional[int] = None,
            criterion: Optional[Criterion] = None
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances from cluster with given index."""


class SimpleClusteredClientManager(ClusteredClientManager):
    """Provides a pool of available clients, grouped by clusters."""

    def num_available_by_cluster(self) -> Dict[str, int]:
        available_dict = {}
        for key, clients in self.clusters.items():
            available_dict[key] = len(clients)
        return available_dict

    def get_default_key(self) -> str:
        return self.default_key

    def num_available_in_cluster(self, cluster_key: str) -> int:
        if cluster_key in self.clusters.keys():
            return len(self.clusters[cluster_key])
        else:
            return 0

    def register_in_cluster(self, client: ClientProxy, cluster_key: str) -> bool:
        if client.cid in self.clients or cluster_key not in self.clusters.keys():
            return False

        self.clients[client.cid] = client
        self.clusters[cluster_key][client.cid] = client
        with self._cv:
            self._cv.notify_all()

        return True

    def all_by_cluster(self) -> Dict[str, Dict[str, ClientProxy]]:
        return self.clusters

    def all_in_cluster(self, cluster_key: str) -> Dict[str, ClientProxy]:
        if cluster_key in self.clusters.keys():
            return self.clusters[cluster_key]
        return {}

    def wait_for_cluster(self, num_clients: int, cluster_key: str, timeout: int = 86400) -> bool:
        with self._cv:
            return self._cv.wait_for(
                lambda: self.num_available_in_cluster(cluster_key) >= num_clients, timeout=timeout
            )

    def sample_cluster(self, num_clients: int, cluster_key: str, min_num_clients: Optional[int] = None,
                       criterion: Optional[Criterion] = None) -> List[ClientProxy]:
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for_cluster(min_num_clients, cluster_key=cluster_key)
        # Sample clients which meet the criterion
        available_cids = list(self.all_in_cluster(cluster_key))
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]

    def __init__(self, seed=None, default_key="A") -> None:
        self.seed = seed
        self.default_key = default_key
        self.clients: Dict[str, ClientProxy] = {}
        self._cv = threading.Condition()
        self.clusters: Dict[str, Dict[str, ClientProxy]] = {default_key: {}}
        self.cluster_partition_ids = {default_key: []}

    def __len__(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        return len(self.clients)

    def num_available(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        return len(self)

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Wait until at least `num_clients` are available.

        Blocks until the requested number of clients is available or until a
        timeout is reached. Current timeout default: 1 day.

        Parameters
        ----------
        num_clients : int
            The number of clients to wait for.
        timeout : int
            The time in seconds to wait for, defaults to 86400 (24h).

        Returns
        -------
        success : bool
        """
        with self._cv:
            return self._cv.wait_for(
                lambda: len(self.clients) >= num_clients, timeout=timeout
            )

    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy

        Returns
        -------
        success : bool
            Indicating if registration was successful. False if ClientProxy is
            already registered or can not be registered for any reason.
        """
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client
        self.clusters[list(self.clusters)[0]][client.cid] = client
        self.cluster_partition_ids[list(self.clusters)[0]] += [len(self.clients)]
        log(WARNING,
            "new cluster keys {}, values {}".format([key for key in self.cluster_partition_ids.keys()], [val for val in self.cluster_partition_ids.values()]))
        with self._cv:
            self._cv.notify_all()

        return True

    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance.

        This method is idempotent.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy
        """
        if client.cid in self.clients:
            del self.clients[client.cid]
            for cluster_key in self.clusters.keys():
                if client.cid in self.clusters[cluster_key]:
                    del self.clusters[cluster_key][client.cid]
            with self._cv:
                self._cv.notify_all()

    def all(self) -> Dict[str, ClientProxy]:
        """Return all available clients."""
        return self.clients

    def _load_cluster(self, filename) -> Dict[str, List[str]]:
        with open(filename) as f:
            cluster_mapping = json.load(f)
            return cluster_mapping

    """    def compute_clusters_data(self) -> bool:
        #TODO
        path = expanduser("~")+"/Pycharm/Exercises/flower/CluteringFL"
        filename = "../../../Statistics/cluster_mapping.json"
        cluster_mapping = self._load_cluster(filename=filename)
        new_clusters = {}
        all_clients = self.all()
        all_clients_assigned = {cid: False for cid in all_clients.keys()}
        for cluster_key, client_list in cluster_mapping.items():
            for client_id in client_list:
                if client_id not in all_clients.keys():
                    raise ValueError("Client {} in clustering output does not exist in the system.".format(client_id))
                if cluster_key not in new_clusters.keys():
                    new_clusters[cluster_key] = {}
                new_clusters[cluster_key][client_id] = all_clients[client_id]
                all_clients_assigned[client_id] = True
        if all(all_clients_assigned.items()):
            self.clusters = new_clusters
            self.cluster_partition_ids = cluster_mapping
            return True
        else:
            log(WARNING, "not all clients were assigned to a new cluster; disregarding new clustering structure.")
            return False"""

    def compute_clusters_data(self) -> bool:
        #TODO call clustering function directly; pass seed
        path = expanduser("~")+"/PycharmProjects/Exercises/flower/CluteringFL"
        filename = path+"/cluster_mapping.json"
        cluster_mapping = self._load_cluster(filename=filename)
        new_clusters = {}
        all_clients = self.all()
        all_clients_keys = list(all_clients.keys())
        all_clients_assigned = {cid: False for cid in all_clients.keys()}
        arbitrary_client_idx = 0
        for cluster_key, client_list in cluster_mapping.items():
            for partition_id in client_list:
                if cluster_key not in new_clusters.keys():
                    new_clusters[cluster_key] = {}
                new_clusters[cluster_key][all_clients_keys[arbitrary_client_idx]] = all_clients[all_clients_keys[arbitrary_client_idx]]
                all_clients_assigned[all_clients_keys[arbitrary_client_idx]] = True
                arbitrary_client_idx += 1
        if all(all_clients_assigned.items()):
            self.clusters = new_clusters
            self.cluster_partition_ids = cluster_mapping
            return True
        else:
            log(WARNING, "not all clients were assigned to a new cluster; disregarding new clustering structure.")
            return False

    def split_cluster(self, cluster_key: str) -> bool:
        """if self.num_available_in_cluster(cluster_key) > 2:
            clients_in_cluster = self.clusters[cluster_key]
            new_cluster_key1 = cluster_key+"1"
            new_cluster_key2 = cluster_key + "2"
            self.clusters[new_cluster_key1] = {}
            self.clusters[new_cluster_key2] = {}
            for idx, client in enumerate(clients_in_cluster.values()):
                if idx % 2 == 0:
                    self.clusters[new_cluster_key1][client.cid] = client
                else:
                    self.clusters[new_cluster_key2][client.cid] = client
            del self.clusters[cluster_key]
            return True
        log(WARNING, "not enough clients available to split cluster {}.".format(cluster_key))
        return False"""
        return False

    def get_cluster_keys(self) -> List[str]:
        return list(self.clusters)

    def get_cluster_partition_ids(self) -> Dict[str, List[int]]:
        return self.cluster_partition_ids

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]
