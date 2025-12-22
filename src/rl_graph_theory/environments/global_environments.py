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


class GlobalSetEnvironment(GraphEnvironment):
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

        self._step_count: Optional[int] = None

    def reset_batch(self, batch_size: int) -> Tuple[np.ndarray, EpisodeStatus]:
        initial_graph_batch = self.initial_graph_generator(batch_size=batch_size)

        if self._flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            format_representation = initial_graph_batch.flattened_row_major
        else:
            format_representation = initial_graph_batch.flattened_clockwise

        if self._edge_colors == 2:
            self._state_batch = format_representation.copy()
        else:
            color_indices = np.arange(1, self._edge_colors, dtype=int)
            temp = (format_representation[:, None, :] == color_indices[:, None]).astype(int)
            self._state_batch = temp.reshape(-1, (self._edge_colors - 1) * self._flattened_length)

        self._status = EpisodeStatus.IN_PROGRESS
        self._step_count = 0

        return self._state_batch, self._status

    def _transition_batch(self, action_batch: np.ndarray) -> None:
        if self._edge_colors == 2:
            rows = np.arange(self._state_batch.shape[0], dtype=int)
            self._state_batch[rows, action_batch[:, 0]] = action_batch[:, 1]
        else:
            temp = self._state_batch.reshape(-1, self._edge_colors - 1, self._flattened_length)
            rows = np.arange(self._state_batch.shape[0], dtype=int)
            temp[rows, :, action_batch[:, 0]] = 0
            temp[rows, action_batch[:, 1] - 1, action_batch[:, 0]] = action_batch[:, 1] != 0

        self._step_count += 1
        if self._step_count >= self.episode_length:
            self._status = EpisodeStatus.TRUNCATED

    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> GraphBatch:
        if self._edge_colors == 2:
            return GraphBatch.from_flattened(
                flattened=state_batch,
                flattened_ordering=self._flattened_ordering,
                edge_colors=self._edge_colors,
                is_directed=self._is_directed,
                allow_loops=self._allow_loops,
            )

        temp = state_batch.reshape(-1, self._edge_colors - 1, self._flattened_length)
        color_indices = np.arange(1, self._edge_colors, dtype=int)
        result = (temp * color_indices[:, None]).sum(axis=1)

        return GraphBatch.from_flattened(
            flattened=result,
            flattened_ordering=self._flattened_ordering,
            edge_colors=self._edge_colors,
            is_directed=self._is_directed,
            allow_loops=self._allow_loops,
        )


class GlobalFlipEnvironment(GraphEnvironment):
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

        self._step_count: Optional[int] = None

    def reset_batch(self, batch_size: int) -> Tuple[np.ndarray, EpisodeStatus]:
        initial_graph_batch = self.initial_graph_generator(batch_size=batch_size)

        if self._flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            format_representation = initial_graph_batch.flattened_row_major
        else:
            format_representation = initial_graph_batch.flattened_clockwise

        self._state_batch = format_representation.copy()
        self._status = EpisodeStatus.IN_PROGRESS
        self._step_count = 0

        return self._state_batch, self._status

    def _transition_batch(self, action_batch: np.ndarray) -> None:
        rows = np.arange(self._state_batch.shape[0], dtype=int)
        if self._flip_only:
            self._state_batch[rows, action_batch[:, 0]] ^= 1
        else:
            self._state_batch[rows, action_batch[:, 0]] ^= action_batch[:, 1]

        self._step_count += 1
        if self._step_count >= self.episode_length:
            self._status = EpisodeStatus.TRUNCATED

    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> GraphBatch:
        return GraphBatch.from_flattened(
            flattened=state_batch,
            flattened_ordering=self._flattened_ordering,
            is_directed=self._is_directed,
            allow_loops=self._allow_loops,
        )
