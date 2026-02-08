"""
This ``Python`` module defines two global reinforcement learning environments that inherit from the
`GraphEnvironment` class. These environments model graph-building games in which the edges (resp.
arcs) are initially fully colored in some predetermined manner, and at each step, any edge (resp.
arc) can be properly recolored with any color.
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


class GlobalSetEnvironment(GraphEnvironment):
    """
    This class inherits from the `GraphEnvironment` class and models a graph-building game in which
    the edges (resp. arcs) are initially fully colored in some manner, and at each step, any edge
    (resp. arc) can be potentially recolored with any color. Users can configure the graph order,
    the number of proper edge colors, whether the graphs are directed or undirected, whether loops
    are allowed, and the mechanism that controls how the initial fully colored graphs are
    generated, which can be deterministic or nondeterministic.

    The RL tasks in this environment are continuing, and the total number of actions to be
    performed, i.e., the episode length, is configurable.

    Each state is represented by a binary `numpy.ndarray` vector of type `numpy.uint8` and length
    ``(_edge_colors - 1) * _flattened_length``, where ``_edge_colors`` is the configured number of
    proper edge colors, and ``_flattened_length`` is the flattened length of the graphs. In the
    state vector, the first ``_flattened_length`` bits indicate which edges (resp. arcs) currently
    have color 1, the next ``_flattened_length`` bits indicate which edges (resp. arcs) currently
    have color 2, and this pattern continues up to color ``_edge_colors - 1``. The edges (resp.
    arcs) are ordered according to the selected `FlattenedOrdering` (row-major or clockwise).

    Each action is represented by a `numpy.int32` integer between 0 and
    ``_edge_colors * _flattened_length - 1``. Given an action number ``a``, the edge (resp. arc) to
    be recolored is determined by ``a % _flattened_length``, and the color to assign is determined
    by ``a // _flattened_length``.

    :ivar _state_batch: See the description of the `GraphEnvironment._state_batch` attribute.
    :ivar _status: See the description of the `GraphEnvironment._status` attribute.
    :ivar _edge_colors: The number of proper edge colors in the graphs to be constructed, given as
        a positive `int` that is at least 2.
    :ivar _is_directed: A `bool` indicating whether the graphs to be constructed are directed or
        undirected.
    :ivar _allow_loops: A `bool` indicating whether loops are allowed in the graphs to be
        constructed.
    :ivar _flattened_ordering: An item of the `FlattenedOrdering` enumeration specifying the edge
        (resp. arc) ordering (row-major or clockwise).
    :ivar initial_graph_generator: A `GraphGenerator` function that defines how the underlying
        fully colored graphs are generated for the initial states. This attribute may be
        reconfigured between independent batches of episodes.
    :ivar _flattened_length: A positive `int` equal to the flattened length of the graphs to be
        constructed.
    :ivar _state_length: A positive `int` equal to ``(_edge_colors - 1) * _flattened_length``, i.e.,
        the length of each state vector.
    :ivar _episode_length: A positive `int` specifying the episode length, i.e., the total number
        of actions in each episode.
    :ivar _step_count: Either `None` or a nonnegative `int` counting how many actions have been
        executed in the current batch of episodes. When `_step_count` equals `_episode_length`, the
        episode has reached a final state. This attribute is updated after each call to
        `reset_batch` or `step_batch`.
    """

    def __init__(
        self,
        graph_invariant: GraphInvariant,
        graph_order: int,
        episode_length: Optional[int] = None,
        flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
        edge_colors: int = 2,
        is_directed: bool = False,
        allow_loops: bool = False,
        initial_graph_generator: Optional[GraphGenerator] = None,
        graph_invariant_diff: Optional[GraphInvariantDiff] = None,
        sparse_setting: bool = False,
    ):
        """
        This constructor initializes an instance of the `GlobalSetEnvironment` class.

        :param graph_invariant: A `GraphInvariant` function that computes the graph invariant
            values associated with a batch of underlying graphs. These values are the quantities to
            be maximized by the environment.
        :param graph_order: A positive `int` (not below 2) that represents the graph order of the
            graphs to be constructed.
        :param episode_length: Either `None`, or a positive `int` specifying the number of actions
            in each episode. If `None`, the episode length defaults to the flattened length of the
            graphs to be constructed. The default value is `None`.
        :param flattened_ordering: An item of the `FlattenedOrdering` enumeration specifying
            whether the edges (resp. arcs) are ordered row-major or clockwise. The default value is
            `FlattenedOrdering.ROW_MAJOR`.
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
        self._state_length: int = (self._edge_colors - 1) * self._flattened_length

        if episode_length is not None:
            self._episode_length: int = episode_length
        # If the ``episode_length`` argument is `None`, then the episode length should match the
        # ``_flattened_length`` attribute.
        else:
            self._episode_length: int = self._flattened_length

        self._step_count: Optional[int] = None

    @property
    def state_length(self) -> int:
        return self._state_length

    @property
    def state_dtype(self) -> np.dtype:
        return np.uint8

    @property
    def action_number(self) -> int:  # pragma: no cover
        return self._edge_colors * self._flattened_length

    @property
    def action_mask(self) -> Optional[np.ndarray]:  # pragma: no cover
        return None

    @property
    def episode_length(self) -> int:  # pragma: no cover
        return self._episode_length

    @episode_length.setter
    def episode_length(self, episode_length: int):  # pragma: no cover
        """
        This setter allows the user to potentially reconfigure the episode length between two
        independent batches of episodes. It should not be used while a batch of episodes is
        currently in progress.

        :param episode_length: A positive `int` specifying the new episode length.
        """

        self._episode_length = episode_length

    @property
    def is_continuing(self) -> bool:
        return True

    def _initialize_batch(self, batch_size: int) -> None:
        # Use the ``initial_graph_generator`` to generate the initial underlying fully colored
        # graphs.
        initial_graph_batch = self.initial_graph_generator(batch_size=batch_size)
        if self._flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            format_representation = initial_graph_batch.flattened_row_major_binary
        else:
            format_representation = initial_graph_batch.flattened_clockwise_binary

        # Initialize the state vectors using the generated underlying fully colored graphs.
        self._state_batch = (
            format_representation[:, -self._edge_colors + 1 :, :].reshape(batch_size, -1).copy()
        )
        self._status = EpisodeStatus.IN_PROGRESS
        self._step_count = 0

    def _transition_batch(self, action_batch: np.ndarray) -> None:
        rows = np.arange(self._state_batch.shape[0])

        recolored_edges = action_batch % self._flattened_length
        new_colors = action_batch // self._flattened_length

        # If the graphs have only two proper edge colors, then the transition can easily be done as
        # follows.
        if self._edge_colors == 2:
            self._state_batch[rows, recolored_edges] = new_colors
        # Otherwise, the next trick should be used.
        else:
            temp = self._state_batch.reshape(-1, self._edge_colors - 1, self._flattened_length)
            temp[rows, :, recolored_edges] = 0
            temp[rows, new_colors - 1, recolored_edges] = new_colors != 0

        self._step_count += 1
        if self._step_count >= self._episode_length:
            self._status = EpisodeStatus.TRUNCATED

    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> Graph:
        return Graph.from_flattened(
            flattened=state_batch.reshape(
                -1, self._edge_colors - 1, self._flattened_length
            ).copy(),
            flattened_ordering=self._flattened_ordering,
            color_representation=ColorRepresentation.BINARY_SLICES,
            edge_colors=self._edge_colors,
            is_directed=self._is_directed,
            allow_loops=self._allow_loops,
        )


class GlobalFlipEnvironment(GraphEnvironment):
    """
    This class inherits from `GraphEnvironment` and models a graph-building game for constructing
    2-edge-colored looped complete graphs. In this environment, all edges (resp. arcs) are
    initially fully colored in some manner, and at each step, any edge (resp. arc) can be
    potentially flipped. Users can configure the graph order, whether the graphs are directed or
    undirected, whether loops are allowed, and the mechanism for generating the initial fully
    colored graphs, which can be deterministic or nondeterministic.

    The RL tasks in this environment are continuing. The total number of actions to be performed,
    i.e., the episode length, is configurable.

    Each state is represented by a binary `numpy.ndarray` vector of type `numpy.uint8` that encodes
    the current graph in one of the two flattened formats with color numbers. The edge ordering
    (row-major or clockwise) is configurable by the user at initialization and remains fixed.

    The environment supports two modes of operation controlled by the `flip_only` parameter. If
    `flip_only` is `False` (the default), each action is a `numpy.int32` integer between 0 and
    ``2 * flattened_length - 1``. In this case, for an action ``a``, ``a % flattened_length`` gives
    the index of the edge (resp. arc) to potentially flip, and ``a // flattened_length`` is a
    binary value indicating whether the edge should actually be flipped (1) or left unchanged (0).
    If `flip_only` is `True`, each action is a `numpy.int32` integer between 0 and
    ``flattened_length - 1``, specifying the index of the edge (resp. arc) that must be flipped.

    :ivar _state_batch: See `GraphEnvironment._state_batch`.
    :ivar _status: See `GraphEnvironment._status`.
    :ivar _is_directed: A `bool` indicating whether the graphs to be constructed are directed or
        undirected.
    :ivar _allow_loops: A `bool` indicating whether loops are allowed in the graphs to be
        constructed.
    :ivar _flip_only: A `bool` specifying whether the selected edges (resp. arcs) must be flipped.
    :ivar _flattened_ordering: An item of the `FlattenedOrdering` enumeration specifying the edge
        (resp. arc) ordering (row-major or clockwise).
    :ivar initial_graph_generator: A `GraphGenerator` function that defines how the underlying
        fully colored graphs are generated for the initial states. This attribute may be
        reconfigured between independent batches of episodes.
    :ivar _flattened_length: A positive `int` equal to the flattened length of the graphs to be
        constructed.
    :ivar _episode_length: A positive `int` specifying the episode length, i.e., the total number
        of actions in each episode.
    :ivar _step_count: Either `None` or a nonnegative `int` counting how many actions have been
        executed in the current batch of episodes. When `_step_count` equals `_episode_length`, the
        episode has reached a final state. This attribute is updated after each call to
        `reset_batch` or `step_batch`.
    """

    def __init__(
        self,
        graph_invariant: GraphInvariant,
        graph_order: int,
        episode_length: Optional[int] = None,
        flip_only: bool = False,
        flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
        is_directed: bool = False,
        allow_loops: bool = False,
        initial_graph_generator: Optional[GraphGenerator] = None,
        graph_invariant_diff: Optional[GraphInvariantDiff] = None,
        sparse_setting: bool = False,
    ):
        """
        This constructor initializes an instance of the `GlobalFlipEnvironment` class.

        :param graph_invariant: A `GraphInvariant` function that computes the graph invariant
            values associated with a batch of underlying graphs. These values are the quantities to
            be maximized by the environment.
        :param graph_order: A positive `int` (not below 2) that represents the graph order of the
            graphs to be constructed.
        :param episode_length: Either `None`, or a positive `int` specifying the number of actions
            in each episode. If `None`, the episode length defaults to the flattened length of the
            graphs to be constructed. The default value is `None`.
        :param flip_only: A `bool` specifying whether the selected edges (resp. arcs) must be
            flipped. The default value is `False`.
        :param flattened_ordering: An item of the `FlattenedOrdering` enumeration specifying
            whether the edges (resp. arcs) are ordered row-major or clockwise. The default value is
            `FlattenedOrdering.ROW_MAJOR`.
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
        self._flip_only: bool = flip_only
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

        if episode_length is not None:
            self._episode_length: int = episode_length
        # If the ``episode_length`` argument is `None`, then the episode length should match the
        # ``_flattened_length`` attribute.
        else:
            self._episode_length: int = self._flattened_length

        self._step_count: Optional[int] = None

    @property
    def state_length(self) -> int:
        return self._flattened_length

    @property
    def state_dtype(self) -> np.dtype:
        return np.uint8

    @property
    def action_number(self) -> int:  # pragma: no cover
        if self._flip_only:
            return self._flattened_length
        else:
            return 2 * self._flattened_length

    @property
    def action_mask(self) -> Optional[np.ndarray]:  # pragma: no cover
        return None

    @property
    def episode_length(self) -> int:  # pragma: no cover
        return self._episode_length

    @episode_length.setter
    def episode_length(self, episode_length: int):  # pragma: no cover
        """
        This setter allows the user to potentially reconfigure the episode length between two
        independent batches of episodes. It should not be used while a batch of episodes is
        currently in progress.

        :param episode_length: A positive `int` specifying the new episode length.
        """

        self._episode_length = episode_length

    @property
    def is_continuing(self) -> bool:
        return True

    def _initialize_batch(self, batch_size: int) -> None:
        # Use the ``initial_graph_generator`` to generate the initial underlying fully colored
        # graphs.
        initial_graph_batch = self.initial_graph_generator(batch_size=batch_size)
        if self._flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            format_representation = initial_graph_batch.flattened_row_major_binary
        else:
            format_representation = initial_graph_batch.flattened_clockwise_binary

        # Initialize the state vectors using the generated underlying fully colored graphs.
        self._state_batch = format_representation[:, -1, :].reshape(batch_size, -1).copy()
        self._status = EpisodeStatus.IN_PROGRESS
        self._step_count = 0

    def _transition_batch(self, action_batch: np.ndarray) -> None:
        rows = np.arange(self._state_batch.shape[0])
        recolored_edges = action_batch % self._flattened_length

        # If the ``_flip_only`` attribute is `True`, then all the selected edges (resp. arcs) must
        # be flipped.
        if self._flip_only:
            self._state_batch[rows, recolored_edges] ^= 1
        # Otherwise, each selected edge (resp. arc) is either flipped or not flipped, depending on
        # the value ``a // flattened_length``.
        else:
            to_flip_or_not_to_flip = action_batch // self._flattened_length
            self._state_batch[rows, recolored_edges] ^= to_flip_or_not_to_flip.astype(np.uint8)

        self._step_count += 1
        if self._step_count >= self._episode_length:
            self._status = EpisodeStatus.TRUNCATED

    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> Graph:
        return Graph.from_flattened(
            flattened=state_batch.reshape(-1, 1, self._flattened_length).copy(),
            flattened_ordering=self._flattened_ordering,
            color_representation=ColorRepresentation.BINARY_SLICES,
            is_directed=self._is_directed,
            allow_loops=self._allow_loops,
        )
