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
    the flattened row-major order or the flattened clockwise order. The user can configure the
    graph order and the number of proper edge colors, as well as choose whether the graphs should
    be directed or undirected, and whether loops should be allowed. The edges (resp. arcs) are
    colored in the same way that a graph is reconstructed from a flattened format. In other words,
    if loops are not allowed, then these edges (resp. arcs) are just deleted and completely
    ignored, and if the graphs are undirected, then only the upper triangular part of an adjacency
    matrix is considered and each edge is essentially colored in both directions at the same time.
    The user can choose whether the edges (resp. arcs) should be colored in the flattened row-major
    order or the flattened clockwise order.

    The RL tasks in this environment are episodic, and the total number of actions to be performed
    equals the number of entries in each of the two flattened graph formats. Therefore, this number
    depends on the selected graph order, on whether the constructed graphs are directed, and on
    whether loops are allowed.

    Each state is represented by a binary vector of length ``edge_colors * flattened_length``,
    where ``edge_colors`` is the configured number of proper edge colors, while
    ``flattened_length`` is the length of the `np.ndarray` list from any of the two flattened graph
    formats of the graphs to be constructed. In this vector, in the first ``flattened_length``
    bits, the value one indicates which of the ``flattened_length`` edges (resp. arcs) has been
    colored with the color 1; in the second ``flattened_length`` bits, the value one indicates
    which of the ``flattened_length`` edges (resp. arcs) has been colored with the color 2; and so
    on until the ``(edge_colors - 1)``-th ``flattened_length`` bits, where the value one indicates
    which of the ``flattened_length`` edges (resp. arcs) has been colored with the color
    ``edge_colors - 1``. The final ``flattened_length`` bits represent a one-hot encoding of the
    position which determines the next edge (resp. arc) to be colored. In other words, there is
    either one value of 1 whose index determines which edge (resp. arc) should be colored next, or
    all the values are 0 and this signifies a terminal state, i.e., a state where all the edges
    (resp. arcs) have been colored. Each action is represented by an integer value between 0 and
    ``edge_colors - 1`` that determines which color the next edge (resp. arc) should be colored
    with.

    :ivar _edge_colors: The number of proper edge colors of the graphs to be constructed.
    :ivar _is_directed: A boolean that indicates whether the graphs to be constructed are a
        $k$-edge-colored looped complete directed graph or a $k$-edge-colored looped complete
        undirected graph.
    :ivar _allow_loops: A boolean that indicates whether the graphs to be constructed are allowed
        to have loops.
    :ivar _flattened_ordering: An item of the `FlattenedOrdering` enumeration that determines
        whether the edges (resp. arcs) should be colored in the flattened row-major order or the
        flattened clockwise order.
    :ivar _flattened_length: A positive integer that determines the length of the `np.ndarray` list
        from any of the two flattened graph formats of the graphs to be constructed. This number
        also represents how many actions are needed in total to reach a terminal state from an
        initial state.
    :ivar _next_entry_index: An integer between 0 and ``_flattened_length`` that signifies the
        index of the next edge (resp. arc) to be colored, in the order determined by the
        ``_flattened_ordering`` attribute. The value of ``_flattened_length`` indicates that all
        the edges (resp. arcs) have been colored and that no edge (resp. arc) should be colored
        next, i.e., a terminal state has been reached.
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
        This constructor initializes an instance of the `IncrementalEnvironment` object.

        :param reward_type: An item of the `RewardType` enumeration that determines the (sub)type
            of reward system to be used in the instantiated environment.
        :param reward_function: The function whose goal is to help compute the rewards in
            accordance with the selected (sub)type of reward system. It plays the role of either
            the ``graph_invariant`` or the ``reward_function`` function from the description of the
            `RewardType` enumeration, and its expected signature varies depending on the
            ``reward_type`` argument.
        :param graph_order: A positive integer (not below two) that represents the graph order of
            the graphs to be constructed.
        :param flattened_ordering: An item of the `FlattenedOrdering` enumeration that determines
            whether the edges (resp. arcs) should be colored in the flattened row-major order or
            the flattened clockwise order. The default value is `FlattenedOrdering.ROW_MAJOR`,
            i.e., the edges (resp. arcs) should be colored in the flattened row-major order by
            default.
        :param edge_colors: A positive integer (not below two) that represents the number of proper
            edge colors of the graphs to be constructed. The default value is two.
        :param is_directed: A boolean that indicates whether the graphs to be constructed are a
            $k$-edge-colored looped complete directed graph or a $k$-edge-colored looped complete
            undirected graph. The default value is `False`, i.e., the graphs to be constructed are
            undirected by default.
        :param allow_loops: A boolean that indicates whether the graphs to be constructed are
            allowed to have loops. The default value is `False`, i.e., the graphs to be constructed
            are not allowed to have loops by default.
        """

        super().__init__(reward_type=reward_type, reward_function=reward_function)

        self._edge_colors: int = edge_colors
        self._is_directed: bool = is_directed
        self._allow_loops: bool = allow_loops
        self._flattened_ordering: FlattenedOrdering = flattened_ordering

        # If the graph is directed...
        if is_directed:
            # If loops are allowed, then count all the adjacency matrix entries.
            if allow_loops:
                self._flattened_length: int = graph_order * graph_order
            # If loops are not allowed, then count the adjacency matrix entries outside the
            # diagonal.
            else:
                self._flattened_length: int = graph_order * (graph_order - 1)
        # If the graph is undirected...
        else:
            # If loops are allowed, then count the entries from the upper triangular part of the
            # adjacency matrix, including the diagonal.
            if allow_loops:
                self._flattened_length: int = graph_order * (graph_order + 1) // 2
            # If loops are not allowed, then count the entries from the upper triangular part of
            # the adjacency matrix, excluding the diagonal.
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
