"""
#TODO
"""

from typing import Callable, Optional, Tuple

import numpy as np

from ..graphs.graph import EdgeOrdering, GraphBatch
from .environment import (
    ActionBatch,
    EpisodeStatus,
    GraphEnvironment,
    RewardType,
    StateBatch,
)


class IncrementalEnvironment(GraphEnvironment):
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
    ):
        """
        #TODO
        """

        super().__init__(reward_type=reward_type, reward_function=reward_function)

        self._graph_size: int = graph_order * (graph_order - 1) // 2
        self._edge_colors: int = edge_colors
        self._edge_ordering: EdgeOrdering = edge_ordering

        self._batch_size: Optional[int] = None
        self._next_edge_index: Optional[int] = None

    def reset_batch(self, batch_size: int) -> Tuple[np.ndarray, EpisodeStatus]:
        self._state_batch = np.zeros((batch_size, self._graph_size * self._edge_colors), dtype=int)
        self._state_batch[:, self._graph_size * (self._edge_colors - 1)] = 1 
        self._status = EpisodeStatus.IN_PROGRESS

        self._batch_size = batch_size
        self._next_edge_index = 0

        return self._state_batch, self._status

    def _transition_batch(self, action_batch: np.ndarray) -> Tuple[np.ndarray, EpisodeStatus]:
        if self._next_edge_index >= self._graph_size:
            raise RuntimeError
        
        rows = np.arange(self._batch_size, dtype=int)
        columns = (action_batch[:, 0] - 1) * self._graph_size + self._next_edge_index

        new_states = self._state_batch.copy()
        new_states[rows, columns] = 1
        new_states[:, self._graph_size * (self._edge_colors - 1) + self._next_edge_index] = 0
        self._next_edge_index += 1

        status = EpisodeStatus.IN_PROGRESS

        if self._next_edge_index < self._graph_size:
            new_states[:, self._graph_size * (self._edge_colors - 1) + self._next_edge_index] = 1
        else:
            status = EpisodeStatus.TERMINATED

        return new_states, status

    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> GraphBatch:
        batch_size = state_batch.shape[0]
        temp = state_batch.reshape(batch_size, self._edge_colors, self._graph_size)
        indices = np.arange(1, self._edge_colors, dtype=int)
        result = (temp[:, :-1, :] * indices[:, None]).sum(axis=1)

        return GraphBatch.from_flattened_batch(
            edge_colors=self._edge_colors,
            edge_ordering=self._edge_ordering,
            flattened_batch=result,
        )
