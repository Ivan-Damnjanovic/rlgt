"""
This ``Python`` module contains the `IncrementalEnvironment` class, which inherits from the
`GraphEnvironment` class and models a graph-building game in which the edges (resp. arcs) are all
initially uncolored and are then colored one by one, either in the flattened row-major order or the
flattened clockwise order.
"""

from typing import Callable, Optional, Tuple

import numpy as np

from ..graphs.graph import FlattenedOrdering, GraphBatch
from .environment import (
    EpisodeStatus,
    GraphEnvironment,
    RewardType,
)


class IncrementalEnvironment(GraphEnvironment):
    """
    This class inherits from the `GraphEnvironment` class and models a graph-building game in which
    the edges (resp. arcs) are all initially uncolored and are then colored one by one, either in
    the flattened row-major order or the flattened clockwise order. The RL tasks in this
    environment are episodic, and the total number of actions to be performed equals the number of
    entries in each of the two flattened graph formats. Therefore, this number depends on the
    selected graph order, on whether the graph is directed, and on whether loops are allowed. The
    number of proper edge colors is also configurable.

    #TODO

    :ivar _edge_colors:
    :ivar _is_directed:
    :ivar _allow_loops:
    :ivar _flattened_ordering:
    :ivar _flattened_length:
    :ivar _next_entry_index:
    """

    def __init__(
        self,
        reward_type: RewardType,
        reward_function: Callable,
        graph_order: int,
        flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
        edge_colors: int = 2,
        is_directed: bool = False,
        allow_loops: bool = False,
    ):
        """
        This constructor initializes an instance of the `IncrementalEnvironment` object with a provided
        (sub)type of reward system and a corresponding function that helps compute the rewards.

        The order must be at least two.

        #TODO
        """

        super().__init__(reward_type=reward_type, reward_function=reward_function)

        self._edge_colors: int = edge_colors
        self._is_directed: bool = is_directed
        self._allow_loops: bool = allow_loops
        self._flattened_ordering: FlattenedOrdering = flattened_ordering

        if is_directed:
            if allow_loops:
                self._flattened_length: int = graph_order * graph_order
            else:
                self._flattened_length: int = graph_order * (graph_order - 1)
        else:
            if allow_loops:
                self._flattened_length: int = graph_order * (graph_order + 1) // 2
            else:
                self._flattened_length: int = graph_order * (graph_order - 1) // 2

        self._next_entry_index: Optional[int] = None

    def reset_batch(self, batch_size: int) -> Tuple[np.ndarray, EpisodeStatus]:
        self._state_batch = np.zeros(
            (batch_size, self._flattened_length * self._edge_colors), dtype=int
        )
        self._state_batch[:, self._flattened_length * (self._edge_colors - 1)] = 1
        self._status = EpisodeStatus.IN_PROGRESS
        self._next_entry_index = 0

        return self._state_batch, self._status

    def _transition_batch(self, action_batch: np.ndarray) -> None:
        rows = np.arange(self._flattened_length, dtype=int)
        columns = (action_batch[:, 0] - 1) * self._flattened_length + self._next_entry_index

        self._state_batch[rows, columns] = 1
        self._state_batch[
            :, self._flattened_length * (self._edge_colors - 1) + self._next_entry_index
        ] = 0
        self._next_entry_index += 1

        if self._next_entry_index < self._flattened_length:
            self._state_batch[
                :, self._flattened_length * (self._edge_colors - 1) + self._next_entry_index
            ] = 1
        else:
            self._status = EpisodeStatus.TERMINATED

    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> GraphBatch:
        temp = state_batch.reshape(-1, self._edge_colors, self._flattened_length)
        indices = np.arange(1, self._edge_colors, dtype=int)
        result = (temp[:, :-1, :] * indices[:, None]).sum(axis=1)

        uncolored_mask = np.maximum.accumulate(temp[:, -1, :])
        result[uncolored_mask.astype(bool)] = self._edge_colors

        return GraphBatch.from_flattened(
            flattened=result,
            flattened_ordering=self._flattened_ordering,
            edge_colors=self._edge_colors,
            is_directed=self._is_directed,
            allow_loops=self._allow_loops,
        )
