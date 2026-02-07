"""
This ``Python`` module defines three linear reinforcement learning environments that inherit from
the `GraphEnvironment` class and model graph-building games where the edges (resp. arcs) are all
initially either uncolored, or fully colored in some predetermined manner, and are then properly
(re)colored one by one, either in the row-major order or the clockwise order.
"""

from typing import Optional

import numpy as np

from ..graphs.graph import Graph
from ..graphs.graph_formats import ColorRepresentation, FlattenedOrdering, GraphFormat
from ..graphs.special_graphs import MonochromaticGraph
from ..graphs.utils import graph_order_to_flattened_length
from .graph_environment import (
    EpisodeStatus,
    GraphEnvironment,
    GraphInvariant,
    GraphInvariantDiff,
)
from .graph_generators import GraphGenerator, create_fixed_graph_generator


class LinearBuildEnvironment(GraphEnvironment):
    """
    This class inherits from the `GraphEnvironment` class and models a graph-building game in which
    the edges (resp. arcs) are initially uncolored and are then properly colored one by one, either
    in row-major order or clockwise order. Users can configure the graph order, the number of
    proper edge colors, whether the graphs are directed or undirected, and whether loops are
    allowed. Edges (resp. arcs) are properly colored according to the reconstruction from a
    flattened format. If loops are not allowed, the corresponding edges are ignored; if the graphs
    are undirected, only the upper-triangular part of the adjacency matrix is considered (including
    or excluding the diagonal depending on whether loops are allowed), and each edge is effectively
    colored in both directions simultaneously.

    The RL tasks in this environment are episodic. The episode length equals the flattened length,
    i.e., the number of entries in any of the two flattened formats with color numbers. This number
    depends on the graph order, on whether the graphs are directed, and on whether loops are
    allowed.

    Each state is represented by a binary `numpy.ndarray` vector of type `numpy.uint8` and length
    ``_edge_colors * _flattened_length``, where ``_edge_colors`` is the configured number of proper
    edge colors and ``_flattened_length`` is the flattened length of the graph. In the state
    vector, the first ``_flattened_length`` bits indicate which edges (resp. arcs) have been
    colored with color 1, the next ``_flattened_length`` bits indicate which edges (resp. arcs)
    have been colored with color 2, and this pattern continues for each color up to
    ``_edge_colors - 1``. The final ``_flattened_length`` bits form a one-hot encoding of the next
    edge (resp. arc) to be properly colored. A value of 1 at a given position indicates which edge
    (resp. arc) is to be colored next, while all zeros signify a terminal state in which all edges
    (resp. arcs) have been properly colored.

    Each action is represented by a `numpy.int32` integer between 0 and ``_edge_colors - 1``, which
    specifies the color to assign to the next edge (resp. arc) in the selected order.

    :ivar _state_batch: See the description of the `GraphEnvironment._state_batch` attribute.
    :ivar _status: See the description of the `GraphEnvironment._status` attribute.
    :ivar _edge_colors: The number of proper edge colors in the graphs to be constructed, given as
        a positive `int` that is at least 2.
    :ivar _is_directed: A `bool` indicating whether the graphs to be constructed are directed or
        undirected.
    :ivar _allow_loops: A `bool` indicating whether loops are allowed in the graphs to be
        constructed.
    :ivar _flattened_ordering: An item of the `FlattenedOrdering` enumeration specifying whether
        the edges (resp. arcs) are colored in row-major or clockwise order.
    :ivar _flattened_length: A positive `int` equal to the flattened length of the graphs to be
        constructed, which also equals the total number of steps needed to reach a terminal state.
    :ivar _state_length: A positive `int` equal to ``_edge_colors * _flattened_length``, i.e., the
        length of each state vector.
    :ivar _step_count: Either `None` or a nonnegative `int` between 0 and `_flattened_length`
        indicating the index of the next edge (resp. arc) to properly color according to
        `_flattened_ordering`. When `_step_count` equals `_flattened_length`, the state is
        terminal. This attribute is initially `None` and updated after each call to `reset_batch`
        or `step_batch`.
    """

    def __init__(
        self,
        graph_invariant: GraphInvariant,
        graph_order: int,
        flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
        edge_colors: int = 2,
        is_directed: bool = False,
        allow_loops: bool = False,
        graph_invariant_diff: Optional[GraphInvariantDiff] = None,
        sparse_setting: bool = False,
    ):
        """
        This constructor initializes an instance of the `LinearBuildEnvironment` class.

        :param graph_invariant: A `GraphInvariant` function that computes the graph invariant
            values associated with a batch of underlying graphs. These values are the quantities to
            be maximized by the environment.
        :param graph_order: A positive `int` (not below 2) that represents the graph order of the
            graphs to be constructed.
        :param flattened_ordering: An item of the `FlattenedOrdering` enumeration indicating
            whether the edges (resp. arcs) should be properly colored in the row-major or clockwise
            order. The default value is `FlattenedOrdering.ROW_MAJOR`.
        :param edge_colors: A positive `int` (not below 2) specifying the number of proper edge
            colors in the graphs to be constructed. The default value is 2.
        :param is_directed: A `bool` indicating whether the graphs to be constructed are directed.
            The default value is `False`.
        :param allow_loops: A `bool` indicating whether loops are allowed in the graphs to be
            constructed. The default value is `False`.
        :param graph_invariant_diff: Either `None`, indicating that graph invariant values are
            always computed directly using ``graph_invariant``, or a `GraphInvariantDiff` function
            that computes element-wise differences of the graph invariant values when the
            environment transitions from one batch of underlying graphs to another. The default
            value is `None`.
        :param sparse_setting: A `bool` indicating whether the sparse setting is enabled. If set to
            `True`, the graph invariant values are computed only for the final batch of actions.
            Otherwise, the graph invariant values are computed after every batch of actions. The
            default value is `False`.
        """

        super().__init__(
            graph_invariant=graph_invariant,
            graph_invariant_diff=graph_invariant_diff,
            sparse_setting=sparse_setting,
        )

        self._edge_colors: int = edge_colors
        self._is_directed: bool = is_directed
        self._allow_loops: bool = allow_loops
        self._flattened_ordering: FlattenedOrdering = flattened_ordering
        self._flattened_length: int = graph_order_to_flattened_length(
            graph_order=graph_order,
            is_directed=is_directed,
            allow_loops=allow_loops,
        )
        self._state_length: int = self._edge_colors * self._flattened_length

        self._step_count: Optional[int] = None

    @property
    def state_length(self) -> int:
        return self._state_length

    @property
    def state_dtype(self) -> np.dtype:
        return np.uint8

    @property
    def action_number(self) -> int:  # pragma: no cover
        return self._edge_colors

    @property
    def action_mask(self) -> Optional[np.ndarray]:  # pragma: no cover
        return None

    @property
    def episode_length(self) -> int:  # pragma: no cover
        return self._flattened_length

    def _initialize_batch(self, batch_size: int) -> None:
        self._state_batch = np.zeros((batch_size, self._state_length), dtype=np.uint8)
        # The zeroth edge (resp. arc) should be properly colored first.
        self._state_batch[:, -self._flattened_length] = 1
        self._status = EpisodeStatus.IN_PROGRESS
        self._step_count = 0

    def _transition_batch(self, action_batch: np.ndarray) -> None:
        # If the graphs have only two proper edge colors, then the transition can easily be done as
        # follows.
        if self._edge_colors == 2:
            self._state_batch[:, self._step_count] = action_batch
        # Otherwise, the next trick should be used.
        else:
            view = self._state_batch.reshape(-1, self._edge_colors, self._flattened_length)
            rows = np.arange(self._state_batch.shape[0])
            view[rows, action_batch - 1, self._step_count] = 1

        # Set the current edge (resp. arc) position flags to zero and increment the ``_step_count``
        # attribute.
        self._state_batch[:, -self._flattened_length + self._step_count] = 0
        self._step_count += 1

        if self._step_count < self._flattened_length:
            # Set the new current edge (resp. arc) position flags to one.
            self._state_batch[:, -self._flattened_length + self._step_count] = 1
        else:
            self._status = EpisodeStatus.TERMINATED

    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> Graph:
        temp = state_batch.reshape(-1, self._edge_colors, self._flattened_length)

        # If all the underlying graphs are fully colored, then extract them in a reduced flattened
        # format with binary slices.
        if np.all(temp[:, -1, :] == 0):
            return Graph.from_flattened(
                flattened=temp[:, :-1, :].copy(),
                flattened_ordering=self._flattened_ordering,
                color_representation=ColorRepresentation.BINARY_SLICES,
                edge_colors=self._edge_colors,
                is_directed=self._is_directed,
                allow_loops=self._allow_loops,
            )

        # Otherwise, extract the underlying graphs in a standard (non-reduced) flattened format
        # with binary slices. First, settle the slices that correspond to the colors $1, 2, \ldots,
        # k - 1$.
        result = np.empty(temp.shape, dtype=np.uint8)
        result[:, 1:, :] = temp[:, :-1, :]

        # Now, settle the slice that corresponds to color 0.
        uncolored_mask = np.maximum.accumulate(temp[:, -1, :], axis=1)
        result[:, 0, :] = 1 - uncolored_mask - np.sum(result[:, 1:, :], axis=1)

        return Graph.from_flattened(
            flattened=result,
            flattened_ordering=self._flattened_ordering,
            color_representation=ColorRepresentation.BINARY_SLICES,
            edge_colors=self._edge_colors,
            is_directed=self._is_directed,
            allow_loops=self._allow_loops,
        )


class LinearSetEnvironment(GraphEnvironment):
    """
    This class inherits from the `GraphEnvironment` class and models a graph-building game in which
    the edges (resp. arcs) are initially fully colored and are then properly recolored one by one,
    either in row-major order or clockwise order. Users can configure the graph order, the number
    of proper edge colors, whether the graphs are directed or undirected, and whether loops are
    allowed. Additionally, the user can configure the mechanism that controls how the initial fully
    colored graphs are generated, which may be deterministic or nondeterministic. Edges (resp.
    arcs) are recolored according to the reconstruction from a flattened format. If loops are not
    allowed, the corresponding edges are ignored; if the graphs are undirected, only the
    upper-triangular part of the adjacency matrix is considered (including or excluding the
    diagonal depending on whether loops are allowed), and each edge is effectively recolored in
    both directions simultaneously.

    The RL tasks in this environment are episodic. The episode length equals the flattened length,
    i.e., the number of entries in any of the two flattened formats with color numbers. This number
    depends on the graph order, on whether the graphs are directed, and on whether loops are
    allowed.

    Each state is represented by a binary `numpy.ndarray` vector of type `numpy.uint8` and length
    ``_edge_colors * _flattened_length``, where ``_edge_colors`` is the configured number of proper
    edge colors and ``_flattened_length`` is the flattened length of the graph. In the state
    vector, the first ``_flattened_length`` bits indicate which edges (resp. arcs) currently
    have color 1, the next ``_flattened_length`` bits indicate which edges (resp. arcs) have
    color 2, and this pattern continues up to ``_edge_colors - 1``. The final ``_flattened_length``
    bits form a one-hot encoding of the next edge (resp. arc) to be properly recolored. A value of
    1 at a given position indicates which edge (resp. arc) is to be recolored next, while all zeros
    signify a terminal state in which all edges (resp. arcs) have been properly recolored.

    Each action is represented by a `numpy.int32` integer between 0 and ``_edge_colors - 1``, which
    specifies the color to assign to the next edge (resp. arc) in the selected order.

    :ivar _state_batch: See the description of the `GraphEnvironment._state_batch` attribute.
    :ivar _status: See the description of the `GraphEnvironment._status` attribute.
    :ivar _edge_colors: The number of proper edge colors in the graphs to be constructed, given as
        a positive `int` that is at least 2.
    :ivar _is_directed: A `bool` indicating whether the graphs to be constructed are directed or
        undirected.
    :ivar _allow_loops: A `bool` indicating whether loops are allowed in the graphs to be
        constructed.
    :ivar _flattened_ordering: An item of the `FlattenedOrdering` enumeration specifying whether
        the edges (resp. arcs) are recolored in row-major or clockwise order.
    :ivar initial_graph_generator: A `GraphGenerator` function that defines how the underlying
        fully colored graphs are generated for the initial states. This attribute may be
        reconfigured between independent batches of episodes.
    :ivar _flattened_length: A positive `int` equal to the flattened length of the graphs to be
        constructed, which also equals the total number of steps needed to reach a terminal state.
    :ivar _state_length: A positive `int` equal to ``_edge_colors * _flattened_length``, i.e., the
        length of each state vector.
    :ivar _step_count: Either `None` or a nonnegative `int` between 0 and `_flattened_length`
        indicating the index of the next edge (resp. arc) to properly recolored according to
        `_flattened_ordering`. When `_step_count` equals `_flattened_length`, the state is
        terminal. This attribute is initially `None` and updated after each call to `reset_batch`
        or `step_batch`.
    """

    def __init__(
        self,
        graph_invariant: GraphInvariant,
        graph_order: int,
        flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
        edge_colors: int = 2,
        is_directed: bool = False,
        allow_loops: bool = False,
        initial_graph_generator: Optional[GraphGenerator] = None,
        graph_invariant_diff: Optional[GraphInvariantDiff] = None,
        sparse_setting: bool = False,
    ):
        """
        This constructor initializes an instance of the `LinearSetEnvironment` class.

        :param graph_invariant: A `GraphInvariant` function that computes the graph invariant
            values associated with a batch of underlying graphs. These values are the quantities to
            be maximized by the environment.
        :param graph_order: A positive `int` (not below 2) that represents the graph order of the
            graphs to be constructed.
        :param flattened_ordering: An item of the `FlattenedOrdering` enumeration indicating
            whether the edges (resp. arcs) should be properly recolored in the row-major or
            clockwise order. The default value is `FlattenedOrdering.ROW_MAJOR`.
        :param edge_colors: A positive `int` (not below 2) specifying the number of proper edge
            colors in the graphs to be constructed. The default value is 2.
        :param is_directed: A `bool` indicating whether the graphs to be constructed are directed.
            The default value is `False`.
        :param allow_loops: A `bool` indicating whether loops are allowed in the graphs to be
            constructed. The default value is `False`.
        :param initial_graph_generator: Either `None` or a `GraphGenerator` function that
            determines how the initial fully colored graphs are generated for the batch of initial
            states. If `None`, all edges (resp. arcs) in all graphs are initially colored with
            color 0. The default value is `None`.
        :param graph_invariant_diff: Either `None`, indicating that graph invariant values are
            always computed directly using ``graph_invariant``, or a `GraphInvariantDiff` function
            that computes element-wise differences of the graph invariant values when the
            environment transitions from one batch of underlying graphs to another. The default
            value is `None`.
        :param sparse_setting: A `bool` indicating whether the sparse setting is enabled. If set to
            `True`, the graph invariant values are computed only for the final batch of actions.
            Otherwise, the graph invariant values are computed after every batch of actions. The
            default value is `False`.
        """

        super().__init__(
            graph_invariant=graph_invariant,
            graph_invariant_diff=graph_invariant_diff,
            sparse_setting=sparse_setting,
        )

        self._edge_colors: int = edge_colors
        self._is_directed: bool = is_directed
        self._allow_loops: bool = allow_loops
        self._flattened_ordering: FlattenedOrdering = flattened_ordering

        if initial_graph_generator is not None:
            self.initial_graph_generator: GraphGenerator = initial_graph_generator
        else:
            # By default, all the edges (resp. arcs) in all the graphs should be colored with color
            # 0.
            graph_format = (
                GraphFormat.FLATTENED_ROW_MAJOR_BINARY
                if flattened_ordering == FlattenedOrdering.ROW_MAJOR
                else GraphFormat.FLATTENED_CLOCKWISE_BINARY
            )
            self.initial_graph_generator: GraphGenerator = create_fixed_graph_generator(
                fixed_graph=MonochromaticGraph(
                    graph_formats={graph_format},
                    graph_order=graph_order,
                    edge_colors=edge_colors,
                    is_directed=is_directed,
                    allow_loops=allow_loops,
                ),
                graph_format=graph_format,
            )

        self._flattened_length: int = graph_order_to_flattened_length(
            graph_order=graph_order,
            is_directed=is_directed,
            allow_loops=allow_loops,
        )
        self._state_length: int = self._edge_colors * self._flattened_length

        self._step_count: Optional[int] = None

    @property
    def state_length(self) -> int:
        return self._state_length

    @property
    def state_dtype(self) -> np.dtype:
        return np.uint8

    @property
    def action_number(self) -> int:  # pragma: no cover
        return self._edge_colors

    @property
    def action_mask(self) -> Optional[np.ndarray]:  # pragma: no cover
        return None

    @property
    def episode_length(self) -> int:  # pragma: no cover
        return self._flattened_length

    def _initialize_batch(self, batch_size: int) -> None:
        # Use the ``initial_graph_generator`` to generate the initial underlying fully colored
        # graphs.
        initial_graph_batch = self.initial_graph_generator(batch_size=batch_size)
        if self._flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            format_representation = initial_graph_batch.flattened_row_major_binary
        else:
            format_representation = initial_graph_batch.flattened_clockwise_binary

        # Initialize the state vectors using the generated underlying fully colored graphs.
        temp = format_representation[:, -self._edge_colors + 1 :, :].reshape(batch_size, -1)
        self._state_batch = np.zeros((batch_size, self._state_length), dtype=np.uint8)
        self._state_batch[:, : -self._flattened_length] = temp
        # The zeroth edge (resp. arc) should be potentially flipped first.
        self._state_batch[:, -self._flattened_length] = 1

        self._status = EpisodeStatus.IN_PROGRESS
        self._step_count = 0

    def _transition_batch(self, action_batch: np.ndarray) -> None:
        # If the graphs have only two proper edge colors, then the transition can easily be done as
        # follows.
        if self._edge_colors == 2:
            self._state_batch[:, self._step_count] = action_batch
        # Otherwise, the next trick should be used.
        else:
            view = self._state_batch.reshape(-1, self._edge_colors, self._flattened_length)
            view[:, :, self._step_count] = 0
            rows = np.arange(self._state_batch.shape[0])
            view[rows, action_batch - 1, self._step_count] = 1

        # Set the current edge (resp. arc) position flags to zero and increment the ``_step_count``
        # attribute.
        self._state_batch[:, -self._flattened_length + self._step_count] = 0
        self._step_count += 1

        if self._step_count < self._flattened_length:
            # Set the new current edge (resp. arc) position flags to one.
            self._state_batch[:, -self._flattened_length + self._step_count] = 1
        else:
            self._status = EpisodeStatus.TERMINATED

    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> Graph:
        result = state_batch.reshape(-1, self._edge_colors, self._flattened_length)[
            :, :-1, :
        ].copy()

        return Graph.from_flattened(
            flattened=result,
            flattened_ordering=self._flattened_ordering,
            color_representation=ColorRepresentation.BINARY_SLICES,
            edge_colors=self._edge_colors,
            is_directed=self._is_directed,
            allow_loops=self._allow_loops,
        )


class LinearFlipEnvironment(GraphEnvironment):
    """
    This class inherits from the `GraphEnvironment` class and models a graph-building game for
    constructing 2-edge-colored looped complete graphs. The edges (resp. arcs) are initially fully
    colored in some manner and can then be potentially flipped one by one in either row-major or
    clockwise order. Users can configure the graph order, whether the graphs are directed or
    undirected, whether loops are allowed, and the mechanism used to generate the initial fully
    colored graphs, which may be deterministic or nondeterministic. The flipping of edges (resp.
    arcs) follows the reconstruction from a flattened format: if loops are not allowed, the
    corresponding edges are ignored, and if the graphs are undirected, only the upper-triangular
    part of the adjacency matrix is considered (including or excluding the diagonal depending on
    loop allowance), so each edge is flipped in both directions simultaneously. The user can select
    whether edges (resp. arcs) are flipped in row-major or clockwise order.

    The RL tasks in this environment are episodic, and the episode length equals the flattened
    length, i.e., the number of entries in either flattened format with color numbers. This length
    depends on the graph order, whether the graphs are directed, and whether loops are allowed.

    Each state is represented by a binary `numpy.ndarray` vector of type `numpy.uint8` and length
    ``2 * _flattened_length``, where ``_flattened_length`` is the flattened length of the graphs.
    In the state vector, the first ``_flattened_length`` bits indicate which edges (resp. arcs) are
    currently colored with color 1, while the final ``_flattened_length`` bits form a one-hot
    encoding of the edge (resp. arc) to be potentially flipped next. A 1 at a given position
    identifies the next edge (resp. arc) to potentially flip, while all zeros indicate a terminal
    state in which all edges (resp. arcs) have been traversed.

    Each action is represented by a `numpy.int32` integer, either 0 or 1. The value 1 flips the
    color of the next edge (resp. arc), while 0 leaves it unchanged.

    :ivar _state_batch: See the description of the `GraphEnvironment._state_batch` attribute.
    :ivar _status: See the description of the `GraphEnvironment._status` attribute.
    :ivar _is_directed: A `bool` indicating whether the graphs to be constructed are directed or
        undirected.
    :ivar _allow_loops: A `bool` indicating whether loops are allowed in the graphs to be
        constructed.
    :ivar _flattened_ordering: An item of the `FlattenedOrdering` enumeration specifying whether
        the edges (resp. arcs) are potentially flipped in row-major or clockwise order.
    :ivar initial_graph_generator: A `GraphGenerator` function that defines how the underlying
        fully colored graphs are generated for the initial states. This attribute may be
        reconfigured between independent batches of episodes.
    :ivar _flattened_length: A positive `int` equal to the flattened length of the graphs to be
        constructed, which also equals the total number of steps needed to reach a terminal state.
    :ivar _state_length: A positive `int` equal to ``_edge_colors * _flattened_length``, i.e., the
        length of each state vector.
    :ivar _step_count: Either `None` or a nonnegative `int` between 0 and `_flattened_length`
        indicating the index of the next edge (resp. arc) to potentially flipped according to
        `_flattened_ordering`. When `_step_count` equals `_flattened_length`, the state is
        terminal. This attribute is initially `None` and updated after each call to `reset_batch`
        or `step_batch`.
    """

    def __init__(
        self,
        graph_invariant: GraphInvariant,
        graph_order: int,
        flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
        is_directed: bool = False,
        allow_loops: bool = False,
        initial_graph_generator: Optional[GraphGenerator] = None,
        graph_invariant_diff: Optional[GraphInvariantDiff] = None,
        sparse_setting: bool = False,
    ):
        """
        This constructor initializes an instance of the `LinearFlipEnvironment` class.

        :param graph_invariant: A `GraphInvariant` function that computes the graph invariant
            values associated with a batch of underlying graphs. These values are the quantities to
            be maximized by the environment.
        :param graph_order: A positive `int` (not below 2) that represents the graph order of the
            graphs to be constructed.
        :param flattened_ordering: An item of the `FlattenedOrdering` enumeration indicating
            whether the edges (resp. arcs) should be potentially flipped in the row-major or
            clockwise order. The default value is `FlattenedOrdering.ROW_MAJOR`.
        :param is_directed: A `bool` indicating whether the graphs to be constructed are directed.
            The default value is `False`.
        :param allow_loops: A `bool` indicating whether loops are allowed in the graphs to be
            constructed. The default value is `False`.
        :param initial_graph_generator: Either `None` or a `GraphGenerator` function that
            determines how the initial fully colored graphs are generated for the batch of initial
            states. If `None`, all edges (resp. arcs) in all graphs are initially colored with
            color 0. The default value is `None`.
        :param graph_invariant_diff: Either `None`, indicating that graph invariant values are
            always computed directly using ``graph_invariant``, or a `GraphInvariantDiff` function
            that computes element-wise differences of the graph invariant values when the
            environment transitions from one batch of underlying graphs to another. The default
            value is `None`.
        :param sparse_setting: A `bool` indicating whether the sparse setting is enabled. If set to
            `True`, the graph invariant values are computed only for the final batch of actions.
            Otherwise, the graph invariant values are computed after every batch of actions. The
            default value is `False`.
        """

        super().__init__(
            graph_invariant=graph_invariant,
            graph_invariant_diff=graph_invariant_diff,
            sparse_setting=sparse_setting,
        )

        self._is_directed: bool = is_directed
        self._allow_loops: bool = allow_loops
        self._flattened_ordering: FlattenedOrdering = flattened_ordering

        if initial_graph_generator is not None:
            self.initial_graph_generator: GraphGenerator = initial_graph_generator
        else:
            # By default, all the edges (resp. arcs) in all the graphs should be colored with color
            # 0.
            graph_format = (
                GraphFormat.FLATTENED_ROW_MAJOR_BINARY
                if flattened_ordering == FlattenedOrdering.ROW_MAJOR
                else GraphFormat.FLATTENED_CLOCKWISE_BINARY
            )
            self.initial_graph_generator: GraphGenerator = create_fixed_graph_generator(
                fixed_graph=MonochromaticGraph(
                    graph_formats={graph_format},
                    graph_order=graph_order,
                    is_directed=is_directed,
                    allow_loops=allow_loops,
                ),
                graph_format=graph_format,
            )

        self._flattened_length: int = graph_order_to_flattened_length(
            graph_order=graph_order,
            is_directed=is_directed,
            allow_loops=allow_loops,
        )

        self._step_count: Optional[int] = None

    @property
    def state_length(self) -> int:
        return 2 * self._flattened_length

    @property
    def state_dtype(self) -> np.dtype:
        return np.uint8

    @property
    def action_number(self) -> int:  # pragma: no cover
        return 2

    @property
    def action_mask(self) -> Optional[np.ndarray]:  # pragma: no cover
        return None

    @property
    def episode_length(self) -> int:  # pragma: no cover
        return self._flattened_length

    def _initialize_batch(self, batch_size: int) -> None:
        # Use the ``initial_graph_generator`` to generate the initial underlying fully colored
        # graphs.
        initial_graph_batch = self.initial_graph_generator(batch_size=batch_size)
        if self._flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            format_representation = initial_graph_batch.flattened_row_major_binary
        else:
            format_representation = initial_graph_batch.flattened_clockwise_binary

        self._state_batch = np.zeros((batch_size, 2 * self._flattened_length), dtype=np.uint8)
        self._state_batch[:, : self._flattened_length] = format_representation[:, -1, :]
        # The zeroth edge (resp. arc) should be potentially flipped first.
        self._state_batch[:, self._flattened_length] = 1

        self._status = EpisodeStatus.IN_PROGRESS
        self._step_count = 0

    def _transition_batch(self, action_batch: np.ndarray) -> None:
        self._state_batch[:, self._step_count] ^= action_batch.astype(np.uint8)
        # Set the current edge (resp. arc) position flags to zero and increment the ``_step_count``
        # attribute.
        self._state_batch[:, self._flattened_length + self._step_count] = 0
        self._step_count += 1

        if self._step_count < self._flattened_length:
            # Set the new current edge (resp. arc) position flags to one.
            self._state_batch[:, self._flattened_length + self._step_count] = 1
        else:
            self._status = EpisodeStatus.TERMINATED

    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> Graph:
        result = state_batch.reshape(-1, 2, self._flattened_length)[:, :-1, :].copy()

        return Graph.from_flattened(
            flattened=result,
            flattened_ordering=self._flattened_ordering,
            color_representation=ColorRepresentation.BINARY_SLICES,
            is_directed=self._is_directed,
            allow_loops=self._allow_loops,
        )
