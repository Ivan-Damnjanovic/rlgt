from typing import Callable, Optional, Tuple

import numpy as np
from graph_environment import (
    ActionBatch,
    GraphEnvironment,
    RewardBatch,
    RewardType,
    StateBatch,
)

from ..graphs.graph import EdgeOrdering, Graph, GraphBatch


class LinearEnvironment(GraphEnvironment):
    """
    #TODO
    """

    def __init__(
        self,
        graph_order: int,
        edge_colors: int,
        edge_ordering: EdgeOrdering,
        reward_type: RewardType,
        reward_function: Callable,
        default_initial_graph: Optional[Graph] = None,
        special_initial_graph: Optional[Graph] = None,
        special_initial_graph_probability: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__(reward_type, reward_function)

        self._graph_order: int = graph_order
        self._edge_colors: int = edge_colors
        self._edge_ordering: EdgeOrdering = edge_ordering
        self._special_initial_graph_probability: float = special_initial_graph_probability

        if default_initial_graph is None:
            temp_matrix = np.zeros((graph_order, edge_colors), dtype=int)
            temp_matrix[:, 0] = (1 << graph_order) - 1
            temp_matrix[:, 0] -= 1 << np.arange(graph_order, dtype=int)
            default_initial_graph = Graph(temp_matrix)
        self._default_initial_graph: Graph = default_initial_graph

        if special_initial_graph is None:
            temp_matrix = np.zeros((graph_order, edge_colors), dtype=int)
            temp_matrix[:, 0] = (1 << graph_order) - 1
            temp_matrix[:, 0] -= 1 << np.arange(graph_order, dtype=int)
            special_initial_graph = Graph(temp_matrix)
        self._special_initial_graph: Graph = special_initial_graph

        self._rng = np.random.default_rng(seed)

    @property
    def special_initial_graph(self) -> Graph:
        """
        #TODO
        """

        return self._special_initial_graph

    @special_initial_graph.setter
    def special_initial_graph(self, special_initial_graph: Graph):
        """
        #TODO
        """

        self._special_initial_graph = special_initial_graph

    def reset_batch(self, batch_size: int) -> StateBatch:
        # self._state_batch = np.tile(self._default_initial_graph, (batch_size, 1, 1))

        temp_matrix = np.tile(self._default_initial_graph, (batch_size, 1, 1))

        generated_values = self._rng.random(size=(batch_size,))
        temp_matrix[generated_values < self._special_initial_graph_probability] = (
            self._special_initial_graph
        )

        return self._state_batch

    def step_batch(self, action_batch: ActionBatch) -> Tuple[StateBatch, RewardBatch]:
        pass


# n (n - 1) / 2 * color_number
# to potice od toga sto prvih n (n - 1) / 2 * (color_number - 1) elemenata oznacava za svaku od prvih color_number - 1
# boja koje su grane obojene njom, koje nisu (binarno), i zadnji deo od n (n - 1) / 2 elemenata je one-hot encoding
# za to koja sledeca grana treba da se drnda
