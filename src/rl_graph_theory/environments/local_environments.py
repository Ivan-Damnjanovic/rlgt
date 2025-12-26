"""
#TODO
"""

from typing import Optional, Tuple

import numpy as np

from ..graphs.graph_batch import GraphBatch
from ..graphs.graph_format import FlattenedOrdering, GraphFormat
from ..graphs.special_graphs import MonochromaticGraph
from .graph_environment import (
    EpisodeStatus,
    GraphEnvironment,
    RewardFunction,
    RewardType,
)
from .graph_generators import GraphGenerator, create_fixed_graph_generator


def compute_edge_indices(
    graph_order: int,
    starting_vertices: np.ndarray,
    ending_vertices: np.ndarray,
    flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
    is_directed: bool = False,
    allow_loops: bool = False,
) -> np.ndarray:
    """
    Docstring for compute_edge_indices

    :param graph_order: Description
    :type graph_order: int
    :param starting_vertices: Description
    :type starting_vertices: np.ndarray
    :param ending_vertices: Description
    :type ending_vertices: np.ndarray
    :param is_directed: Description
    :type is_directed: bool
    :param allow_loops: Description
    :type allow_loops: bool
    :return: Description
    :rtype: ndarray[_AnyShape, dtype[Any]]
    """

    if is_directed:
        if flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            if allow_loops:
                result = starting_vertices * graph_order + ending_vertices
            else:
                result = (
                    starting_vertices * (graph_order - 1)
                    + ending_vertices
                    - (ending_vertices >= starting_vertices).astype(int)
                )
        else:
            layer = np.maximum(starting_vertices, ending_vertices)

            if allow_loops:
                result = layer * layer + layer - ending_vertices + starting_vertices
            else:
                result = (
                    layer * layer
                    - ending_vertices
                    + starting_vertices
                    - (ending_vertices <= starting_vertices).astype(int)
                )

    else:
        rows = np.minimum(starting_vertices, ending_vertices)
        columns = np.maximum(starting_vertices, ending_vertices)

        if flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            if allow_loops:
                result = rows * (2 * graph_order - 1 - rows) // 2 + columns
            else:
                result = rows * (2 * graph_order - 3 - rows) // 2 + columns - 1
        else:
            if allow_loops:
                result = columns * (columns + 1) // 2 + rows
            else:
                result = columns * (columns - 1) // 2 + rows

    return result


class LocalSetEnvironment(GraphEnvironment):
    """
    #TODO
    """

    def __init__(
        self,
        reward_type: RewardType,
        reward_function: RewardFunction,
        graph_order: int,
        episode_length: Optional[int] = None,
        flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
        edge_colors: int = 2,
        is_directed: bool = False,
        allow_loops: bool = False,
        initial_graph_generator: Optional[GraphGenerator] = None,
    ):
        """
        #TODO
        """

        super().__init__(reward_type=reward_type, reward_function=reward_function)

        self._edge_colors: int = edge_colors
        self._is_directed: bool = is_directed
        self._allow_loops: bool = allow_loops
        self._graph_order: int = graph_order
        self._flattened_ordering: FlattenedOrdering = flattened_ordering

        if initial_graph_generator is not None:
            self.initial_graph_generator: GraphGenerator = initial_graph_generator
        else:
            graph_format = (
                GraphFormat.FLATTENED_ROW_MAJOR
                if flattened_ordering == FlattenedOrdering.ROW_MAJOR
                else GraphFormat.FLATTENED_CLOCKWISE
            )
            self.initial_graph_generator: GraphGenerator = create_fixed_graph_generator(
                fixed_graph=MonochromaticGraph(
                    graph_format=graph_format,
                    order=graph_order,
                    edge_colors=edge_colors,
                    is_directed=is_directed,
                    allow_loops=allow_loops,
                ),
                graph_format=graph_format,
            )

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

        if episode_length is not None:
            self.episode_length: int = episode_length
        else:
            self.episode_length: int = self._flattened_length

        self._current_vertices: Optional[np.ndarray] = None
        self._step_count: Optional[int] = None

    def reset_batch(self, batch_size: int) -> Tuple[np.ndarray, EpisodeStatus]:
        initial_graph_batch = self.initial_graph_generator(batch_size=batch_size)

        if self._flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            format_representation = initial_graph_batch.flattened_row_major
        else:
            format_representation = initial_graph_batch.flattened_clockwise

        self._state_batch = np.zeros(
            (
                batch_size,
                (self._edge_colors - 1) * self._flattened_length + self._graph_order,
            ),
            dtype=int,
        )
        self._state_batch[:, (self._edge_colors - 1) * self._flattened_length] = 1

        if self._edge_colors == 2:
            self._state_batch[:, : self._flattened_length] = format_representation
        else:
            color_indices = np.arange(1, self._edge_colors, dtype=int)
            temp = (format_representation[:, None, :] == color_indices[:, None]).astype(
                int
            )
            self._state_batch[:, : (self._edge_colors - 1) * self._flattened_length] = (
                temp.reshape(-1, (self._edge_colors - 1) * self._flattened_length)
            )

        self._current_vertices = np.zeros((batch_size,), dtype=int)
        self._status = EpisodeStatus.IN_PROGRESS
        self._step_count = 0

        return self._state_batch, self._status

    def _transition_batch(self, action_batch: np.ndarray) -> None:
        if not self._allow_loops:
            if np.any(self._current_vertices == action_batch[:, 0]):
                raise RuntimeError

        edge_indices = compute_edge_indices(
            graph_order=self._graph_order,
            starting_vertices=self._current_vertices,
            ending_vertices=action_batch[:, 0],
            flattened_ordering=self._flattened_ordering,
            is_directed=self._is_directed,
            allow_loops=self._allow_loops,
        )

        if self._edge_colors == 2:
            rows = np.arange(self._state_batch.shape[0], dtype=int)
            self._state_batch[rows, edge_indices] = action_batch[:, 1]
        else:
            temp = self._state_batch[
                :, : (self._edge_colors - 1) * self._flattened_length
            ].reshape(-1, self._edge_colors - 1, self._flattened_length)
            rows = np.arange(self._state_batch.shape[0], dtype=int)
            temp[rows, :, edge_indices] = 0
            temp[rows, action_batch[:, 1] - 1, edge_indices] = action_batch[:, 1] != 0

        self._state_batch[
            rows,
            self._flattened_length * (self._edge_colors - 1) + self._current_vertices,
        ] = 0
        self._state_batch[
            rows, self._flattened_length * (self._edge_colors - 1) + action_batch[:, 0]
        ] = 1

        self._current_vertices = action_batch[:, 0]
        self._step_count += 1
        if self._step_count >= self.episode_length:
            self._status = EpisodeStatus.TRUNCATED

    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> GraphBatch:
        if self._edge_colors == 2:
            return GraphBatch.from_flattened(
                flattened=state_batch[:, : self._flattened_length],
                flattened_ordering=self._flattened_ordering,
                is_directed=self._is_directed,
                allow_loops=self._allow_loops,
            )

        temp = state_batch[
            :, : (self._edge_colors - 1) * self._flattened_length
        ].reshape(-1, self._edge_colors - 1, self._flattened_length)
        color_indices = np.arange(1, self._edge_colors, dtype=int)
        result = (temp * color_indices[:, None]).sum(axis=1)

        return GraphBatch.from_flattened(
            flattened=result,
            flattened_ordering=self._flattened_ordering,
            edge_colors=self._edge_colors,
            is_directed=self._is_directed,
            allow_loops=self._allow_loops,
        )


class LocalFlipEnvironment(GraphEnvironment):
    """
    #TODO
    """

    def __init__(
        self,
        reward_type: RewardType,
        reward_function: RewardFunction,
        graph_order: int,
        episode_length: Optional[int] = None,
        flip_only: bool = False,
        flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
        is_directed: bool = False,
        allow_loops: bool = False,
        initial_graph_generator: Optional[GraphGenerator] = None,
    ):
        """
        #TODO
        """

        super().__init__(reward_type=reward_type, reward_function=reward_function)

        self._is_directed: bool = is_directed
        self._allow_loops: bool = allow_loops
        self._graph_order: int = graph_order
        self._flip_only: bool = flip_only
        self._flattened_ordering: FlattenedOrdering = flattened_ordering

        if initial_graph_generator is not None:
            self.initial_graph_generator: GraphGenerator = initial_graph_generator
        else:
            graph_format = (
                GraphFormat.FLATTENED_ROW_MAJOR
                if flattened_ordering == FlattenedOrdering.ROW_MAJOR
                else GraphFormat.FLATTENED_CLOCKWISE
            )
            self.initial_graph_generator: GraphGenerator = create_fixed_graph_generator(
                fixed_graph=MonochromaticGraph(
                    graph_format=graph_format,
                    order=graph_order,
                    is_directed=is_directed,
                    allow_loops=allow_loops,
                ),
                graph_format=graph_format,
            )

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

        if episode_length is not None:
            self.episode_length: int = episode_length
        else:
            self.episode_length: int = self._flattened_length

        self._current_vertices: Optional[np.ndarray] = None
        self._step_count: Optional[int] = None

    def reset_batch(self, batch_size: int) -> Tuple[np.ndarray, EpisodeStatus]:
        initial_graph_batch = self.initial_graph_generator(batch_size=batch_size)

        if self._flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            format_representation = initial_graph_batch.flattened_row_major
        else:
            format_representation = initial_graph_batch.flattened_clockwise

        self._state_batch = np.zeros(
            (batch_size, self._flattened_length + self._graph_order), dtype=int
        )
        self._state_batch[:, self._flattened_length] = 1
        self._state_batch[:, : self._flattened_length] = format_representation

        self._current_vertices = np.zeros((batch_size,), dtype=int)
        self._status = EpisodeStatus.IN_PROGRESS
        self._step_count = 0

        return self._state_batch, self._status

    def _transition_batch(self, action_batch: np.ndarray) -> None:
        if not self._allow_loops:
            if np.any(self._current_vertices == action_batch[:, 0]):
                raise RuntimeError

        edge_indices = compute_edge_indices(
            graph_order=self._graph_order,
            starting_vertices=self._current_vertices,
            ending_vertices=action_batch[:, 0],
            flattened_ordering=self._flattened_ordering,
            is_directed=self._is_directed,
            allow_loops=self._allow_loops,
        )

        rows = np.arange(self._state_batch.shape[0], dtype=int)
        if self._flip_only:
            self._state_batch[rows, edge_indices] ^= 1
        else:
            self._state_batch[rows, edge_indices] ^= action_batch[:, 1]

        self._state_batch[rows, self._flattened_length + self._current_vertices] = 0
        self._state_batch[rows, self._flattened_length + action_batch[:, 0]] = 1

        self._current_vertices = action_batch[:, 0]
        self._step_count += 1
        if self._step_count >= self.episode_length:
            self._status = EpisodeStatus.TRUNCATED

    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> GraphBatch:
        return GraphBatch.from_flattened(
            flattened=state_batch[:, : self._flattened_length],
            flattened_ordering=self._flattened_ordering,
            is_directed=self._is_directed,
            allow_loops=self._allow_loops,
        )
