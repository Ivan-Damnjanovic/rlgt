"""
This ``Python`` module defines two local reinforcement learning environments that inherit from the
`GraphEnvironment` class and model graph-building games in which the edges (resp. arcs) are
initially fully colored in some predetermined manner. In these environments, the agent moves
between vertices according to a specified traversal strategy, thereby traversing existing edges
(resp. arcs) and properly recoloring them as part of the interaction process.
"""

from typing import Optional

import numpy as np

from ..graphs.graph import Graph
from ..graphs.graph_formats import ColorRepresentation, FlattenedOrdering, GraphFormat
from ..graphs.special_graphs import MonochromaticGraph
from ..graphs.utils import compute_edge_indices, graph_order_to_flattened_length
from .graph_environment import (
    EpisodeStatus,
    GraphEnvironment,
    GraphInvariant,
    GraphInvariantDiff,
)
from .graph_generators import GraphGenerator, create_fixed_graph_generator


class LocalSetEnvironment(GraphEnvironment):
    """
    This class inherits from the `GraphEnvironment` class and models a graph-building game in which
    the edges (resp. arcs) are initially fully colored in some manner, and an agent moves between
    vertices, thereby traversing existing edges (resp. arcs) and properly recoloring them. More
    precisely, at each step the agent is located at a vertex and selects an edge incident to this
    vertex (resp. an arc starting at this vertex), traverses it, and moves to the other endpoint.
    During traversal, the selected edge (resp. arc) is properly recolored with a chosen color. The
    user can select the graph order, the number of proper edge colors, whether the graphs are
    directed or undirected, and whether loops are allowed. Additionally, the mechanism controlling
    how the initial fully colored graphs are generated can be configured and may be deterministic
    or nondeterministic. The user can also select the vertex at which the agent starts the
    recoloring procedure.

    The RL tasks in this environment are continuing, and the total number of actions to be
    performed, i.e., the episode length, is configurable.

    Each state is represented by a binary `numpy.ndarray` of type `numpy.uint8` and length
    ``(edge_colors - 1) * flattened_length + graph_order``, where ``edge_colors`` is the configured
    number of proper edge colors, ``graph_order`` is the configured graph order, and
    ``flattened_length`` is the flattened length of the graphs to be constructed. In the state
    vector, the first ``flattened_length`` bits indicate which edges (resp. arcs) are currently of
    color 1, the next ``flattened_length`` bits indicate which edges (resp. arcs) are currently of
    color 2, and this pattern continues up to color ``edge_colors - 1``. The ordering of edges
    (resp. arcs) within these blocks is determined by the selected flattened ordering, which can be
    either row-major or clockwise as specified by the `FlattenedOrdering` enumeration. The final
    ``graph_order`` bits form a one-hot encoding that specifies the vertex at which the agent is
    currently located.

    Each action is represented by a `numpy.int32` integer between 0 and
    ``edge_colors * graph_order - 1``. If the action value is ``a``, then ``a % graph_order``
    determines the vertex to which the agent should move from its current location, while
    ``a // graph_order`` determines the color with which the traversed edge (resp. arc) is properly
    recolored. If loops are not allowed, then actions that would keep the agent at the current
    vertex are invalid.

    :ivar _state_batch: See the description of the `GraphEnvironment._state_batch` attribute.
    :ivar _status: See the description of the `GraphEnvironment._status` attribute.
    :ivar _edge_colors: The number of proper edge colors in the graphs to be constructed, given as
        a positive `int` that is at least 2.
    :ivar _is_directed: A `bool` indicating whether the graphs to be constructed are directed or
        undirected.
    :ivar _allow_loops: A `bool` indicating whether loops are allowed in the graphs to be
        constructed.
    :ivar _graph_order: A positive `int` that describes the order of the graphs to be constructed.
    :ivar _flattened_ordering: An item of the `FlattenedOrdering` enumeration specifying the edge
        (resp. arc) ordering (row-major or clockwise).
    :ivar initial_graph_generator: A `GraphGenerator` function that defines how the underlying
        fully colored graphs are generated for the initial states. This attribute may be
        reconfigured between independent batches of episodes.
    :ivar starting_vertex: A nonnegative `int` below the configured graph order that determines
        the vertex at which the agent should start the recoloring procedure. This attribute may be
        reconfigured between independent batches of episodes.
    :ivar _flattened_length: A positive `int` equal to the flattened length of the graphs to be
        constructed.
    :ivar _state_length: A positive `int` that determines the length of each of the state vectors,
        i.e., the number ``(_edge_colors - 1) * _flattened_length + _graph_order``.
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
        starting_vertex: int = 0,
        graph_invariant_diff: Optional[GraphInvariantDiff] = None,
        sparse_setting: bool = False,
    ):
        """
        This constructor initializes an instance of the `LocalSetEnvironment` class.

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
        :param starting_vertex: A nonnegative `int` strictly less than `graph_order` specifying the
            vertex at which the agent starts the recoloring procedure. The default value is 0.
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
        self._graph_order: int = graph_order
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

        self.starting_vertex: int = starting_vertex

        self._flattened_length: int = graph_order_to_flattened_length(
            graph_order=graph_order,
            is_directed=is_directed,
            allow_loops=allow_loops,
        )
        self._state_length: int = (
            self._edge_colors - 1
        ) * self._flattened_length + self._graph_order

        if episode_length is not None:
            self._episode_length: int = episode_length
        # If the ``episode_length`` argument is `None`, then the episode length should match the
        # ``_flattened_length`` attribute.
        else:
            self._episode_length: int = self._flattened_length

        self._current_vertices: Optional[np.ndarray] = None
        self._step_count: Optional[int] = None

    @property
    def state_length(self) -> int:
        return self._state_length

    @property
    def state_dtype(self) -> np.dtype:
        return np.uint8

    @property
    def action_number(self) -> int:  # pragma: no cover
        return self._edge_colors * self._graph_order

    @property
    def action_mask(self) -> Optional[np.ndarray]:
        if self._status is None or self._status != EpisodeStatus.IN_PROGRESS or self._allow_loops:
            return None

        result = np.ones(
            (self._state_batch.shape[0], self._edge_colors, self._graph_order), dtype=bool
        )
        rows = np.arange(self._state_batch.shape[0])
        result[rows, :, self._current_vertices] = False

        return result.reshape(self._state_batch.shape[0], -1)

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

    def _initialize_batch(self, batch_size: int) -> None:
        # Use the ``initial_graph_generator`` to generate the initial underlying fully colored
        # graphs.
        initial_graph_batch = self.initial_graph_generator(batch_size=batch_size)
        if self._flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            format_representation = initial_graph_batch.flattened_row_major_binary
        else:
            format_representation = initial_graph_batch.flattened_clockwise_binary

        # Initialize the state vectors using the generated underlying fully colored graphs. Also,
        # make sure that the current vertex position flags are all set to the configured starting
        # vertex.
        self._state_batch = np.zeros((batch_size, self._state_length), dtype=np.uint8)
        self._state_batch[:, : -self._graph_order] = format_representation[
            :, -self._edge_colors + 1 :, :
        ].reshape(batch_size, -1)
        self._state_batch[:, -self._graph_order + self.starting_vertex] = 1

        self._current_vertices = np.full((batch_size,), self.starting_vertex, dtype=np.int32)
        self._status = EpisodeStatus.IN_PROGRESS
        self._step_count = 0

    def _transition_batch(self, action_batch: np.ndarray) -> None:
        """
        This method applies a batch of actions to the current batch of states and updates the
        `_state_batch` and `_status` attributes to reflect the resulting states and the updated
        batch status.

        :param action_batch: A one-dimensional `numpy.ndarray` of type `numpy.int32` containing the
            actions to be applied. The length of ``action_batch`` must match the number of states
            in ``_state_batch``.

        :note: If loops are not allowed and an action attempts to traverse a loop, a `RuntimeError`
            is raised.
        """

        new_vertices = action_batch % self._graph_order
        new_colors = action_batch // self._graph_order

        # If the agent attempts to traverse a nonexisting loop, a `RuntimeError` will be raised.
        if not self._allow_loops:
            if np.any(self._current_vertices == new_vertices):
                raise RuntimeError("Cannot traverse a nonexisting loop.")

        # Use the ``compute_edge_indices`` auxiliary function to compute the index of the edge
        # (resp. arc) that should be traversed in each of the episodes run in parallel.
        edge_indices = compute_edge_indices(
            graph_order=self._graph_order,
            starting_vertices=self._current_vertices,
            ending_vertices=new_vertices,
            flattened_ordering=self._flattened_ordering,
            is_directed=self._is_directed,
            allow_loops=self._allow_loops,
        )

        rows = np.arange(self._state_batch.shape[0])

        # If the graphs have only two proper edge colors, then the transition can easily be done as
        # follows.
        if self._edge_colors == 2:
            self._state_batch[rows, edge_indices] = new_colors
        # Otherwise, the next trick should be used.
        else:
            temp = self._state_batch[:, : -self._graph_order].reshape(
                -1, self._edge_colors - 1, self._flattened_length
            )
            temp[rows, :, edge_indices] = 0
            temp[rows, new_colors - 1, edge_indices] = new_colors != 0
            self._state_batch[:, : -self._graph_order] = temp.reshape(temp.shape[0], -1)

        # Update the current vertex position flags.
        self._state_batch[rows, self._current_vertices - self._graph_order] = 0
        self._current_vertices = new_vertices
        self._state_batch[rows, self._current_vertices - self._graph_order] = 1

        self._step_count += 1
        if self._step_count >= self._episode_length:
            self._status = EpisodeStatus.TRUNCATED

    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> Graph:
        result = (
            state_batch[:, : -self._graph_order]
            .reshape(-1, self._edge_colors - 1, self._flattened_length)
            .copy()
        )

        return Graph.from_flattened(
            flattened=result,
            flattened_ordering=self._flattened_ordering,
            color_representation=ColorRepresentation.BINARY_SLICES,
            edge_colors=self._edge_colors,
            is_directed=self._is_directed,
            allow_loops=self._allow_loops,
        )


class LocalFlipEnvironment(GraphEnvironment):
    """
    This class inherits from the `GraphEnvironment` class and models a graph-building game used for
    constructing 2-edge-colored looped complete graphs in which the edges (resp. arcs) are
    initially fully colored in some manner, and the agent moves from one vertex to another, thereby
    traversing existing edges (resp. arcs) and potentially flipping them. More precisely, in each
    step, the agent is located at a vertex and selects an incident edge (resp. an outgoing arc),
    traverses it, and moves to the opposite endpoint. While traversing an edge (resp. arc), the
    agent may flip its proper edge color. The user can select the graph order, choose whether the
    graphs are directed or undirected, and specify whether loops are allowed. Additionally, the
    user can configure the mechanism that controls how the initial fully colored graphs are
    generated, which may be either deterministic or nondeterministic. The user can also select the
    vertex at which the agent starts the potential flipping procedure.

    The RL tasks in this environment are continuing, and the total number of actions to be
    performed, i.e., the episode length, is configurable.

    Each state is represented by a binary `numpy.ndarray` of type `numpy.uint8` and length
    ``flattened_length + graph_order``, where ``graph_order`` is the configured graph order and
    ``flattened_length`` is the flattened length of the graphs to be constructed. In the state
    vector, the first ``flattened_length`` bits indicate which of the ``flattened_length`` edges
    (resp. arcs) are currently of color 1. The edges (resp. arcs) are ordered according to the
    selected flattened ordering, which may be either row-major or clockwise. The final
    ``graph_order`` bits form a one-hot encoding of the vertex at which the agent is currently
    located.

    The environment supports two action-handling modes. If ``flip_only`` is set to `False` (the
    default), each action is represented by a `numpy.int32` integer between 0 and
    ``2 * graph_order - 1``. For an action value ``a``, the quantity ``a % graph_order`` specifies
    the vertex to which the agent moves, while ``a // graph_order`` determines whether the
    traversed edge (resp. arc) is flipped. A value of 1 indicates that the proper edge color is
    changed, whereas 0 indicates that it remains unchanged. If ``flip_only`` is set to `True`,
    then every traversed edge (resp. arc) must be flipped. In this case, each action is a
    `numpy.int32` integer between 0 and ``graph_order - 1`` specifying the destination vertex, with
    the traversed edge (resp. arc) necessarily flipped. If loops are not allowed, then the agent
    cannot move to its current vertex.

    :ivar _state_batch: See the description of the `GraphEnvironment._state_batch` attribute.
    :ivar _status: See the description of the `GraphEnvironment._status` attribute.
    :ivar _is_directed: A `bool` indicating whether the graphs to be constructed are directed or
        undirected.
    :ivar _allow_loops: A `bool` indicating whether loops are allowed in the graphs to be
        constructed.
    :ivar _graph_order: A positive `int` that describes the order of the graphs to be constructed.
    :ivar _flip_only: A `bool` indicating whether all traversed edges (resp. arcs) must be flipped.
    :ivar _flattened_ordering: An item of the `FlattenedOrdering` enumeration specifying the edge
        (resp. arc) ordering (row-major or clockwise).
    :ivar initial_graph_generator: A `GraphGenerator` function that defines how the underlying
        fully colored graphs are generated for the initial states. This attribute may be
        reconfigured between independent batches of episodes.
    :ivar starting_vertex: A nonnegative `int` below the configured graph order that determines
        the vertex at which the agent should start the potential flipping procedure. This attribute
        may be reconfigured between independent batches of episodes.
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
        starting_vertex: int = 0,
        graph_invariant_diff: Optional[GraphInvariantDiff] = None,
        sparse_setting: bool = False,
    ):
        """
        This constructor initializes an instance of the `LocalFlipEnvironment` class.

        :param graph_invariant: A `GraphInvariant` function that computes the graph invariant
            values associated with a batch of underlying graphs. These values are the quantities to
            be maximized by the environment.
        :param graph_order: A positive `int` (not below 2) that represents the graph order of the
            graphs to be constructed.
        :param episode_length: Either `None`, or a positive `int` specifying the number of actions
            in each episode. If `None`, the episode length defaults to the flattened length of the
            graphs to be constructed. The default value is `None`.
        :param flip_only: A `bool` specifying whether the traversed edges (resp. arcs) must be
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
        :param starting_vertex: A nonnegative `int` strictly less than `graph_order` specifying the
            vertex at which the agent starts the potential flipping procedure. The default value is
            0.
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
        self._graph_order: int = graph_order
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

        self.starting_vertex: int = starting_vertex

        self._flattened_length: int = graph_order_to_flattened_length(
            graph_order=graph_order,
            is_directed=is_directed,
            allow_loops=allow_loops,
        )

        if episode_length is not None:
            self.episode_length: int = episode_length
        # If the ``episode_length`` argument is `None`, then the episode length should match the
        # ``_flattened_length`` attribute.
        else:
            self.episode_length: int = self._flattened_length

        self._current_vertices: Optional[np.ndarray] = None
        self._step_count: Optional[int] = None

    @property
    def state_length(self) -> int:
        return self._flattened_length + self._graph_order

    @property
    def state_dtype(self) -> np.dtype:
        return np.uint8

    @property
    def action_number(self) -> int:  # pragma: no cover
        if self._flip_only:
            return self._graph_order
        else:
            return 2 * self._graph_order

    @property
    def action_mask(self) -> Optional[np.ndarray]:
        if self._status is None or self._status != EpisodeStatus.IN_PROGRESS or self._allow_loops:
            return None

        rows = np.arange(self._state_batch.shape[0])

        if self._flip_only:
            result = np.ones((self._state_batch.shape[0], self._graph_order), dtype=bool)
            result[rows, self._current_vertices] = False

            return result

        result = np.ones((self._state_batch.shape[0], 2 * self._graph_order), dtype=bool)
        result[rows, self._current_vertices] = False
        result[rows, self._graph_order + self._current_vertices] = False

        return result

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

    def _initialize_batch(self, batch_size: int) -> None:
        # Use the ``initial_graph_generator`` to generate the initial underlying fully colored
        # graphs.
        initial_graph_batch = self.initial_graph_generator(batch_size=batch_size)
        if self._flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            format_representation = initial_graph_batch.flattened_row_major_binary
        else:
            format_representation = initial_graph_batch.flattened_clockwise_binary

        # Initialize the state vectors using the generated underlying fully colored graphs. Also,
        # make sure that the current vertex position flags are all set to the configured starting
        # vertex.
        self._state_batch = np.zeros(
            (batch_size, self._flattened_length + self._graph_order), dtype=np.uint8
        )
        self._state_batch[:, : self._flattened_length] = format_representation[:, -1, :]
        self._state_batch[:, self._flattened_length + self.starting_vertex] = 1

        self._current_vertices = np.full((batch_size,), self.starting_vertex, dtype=np.int32)
        self._status = EpisodeStatus.IN_PROGRESS
        self._step_count = 0

    def _transition_batch(self, action_batch: np.ndarray) -> None:
        """
        This method applies a batch of actions to the current batch of states and updates the
        `_state_batch` and `_status` attributes to reflect the resulting states and the updated
        batch status.

        :param action_batch: A one-dimensional `numpy.ndarray` of type `numpy.int32` containing the
            actions to be applied. The length of ``action_batch`` must match the number of states
            in ``_state_batch``.

        :note: If loops are not allowed and an action attempts to traverse a loop, a `RuntimeError`
            is raised.
        """

        new_vertices = action_batch % self._graph_order

        # If the agent attempts to traverse a nonexisting loop, a `RuntimeError` will be raised.
        if not self._allow_loops:
            if np.any(self._current_vertices == new_vertices):
                raise RuntimeError("Cannot traverse a nonexisting loop.")

        # Use the ``compute_edge_indices`` auxiliary function to compute the index of the edge
        # (resp. arc) that should be traversed in each of the episodes run in parallel.
        edge_indices = compute_edge_indices(
            graph_order=self._graph_order,
            starting_vertices=self._current_vertices,
            ending_vertices=new_vertices,
            flattened_ordering=self._flattened_ordering,
            is_directed=self._is_directed,
            allow_loops=self._allow_loops,
        )

        rows = np.arange(self._state_batch.shape[0], dtype=np.int32)
        # If the ``_flip_only`` attribute is `True`, then all the traversed edges (resp. arcs) must
        # be flipped.
        if self._flip_only:
            self._state_batch[rows, edge_indices] ^= 1
        # Otherwise, each traversed edge (resp. arc) is either flipped or not flipped, depending on
        # the second entry from the corresponding action, which is binary.
        else:
            to_flip_or_not_to_flip = action_batch // self._graph_order
            self._state_batch[rows, edge_indices] ^= to_flip_or_not_to_flip.astype(np.uint8)

        # Update the current vertex position flags.
        self._state_batch[rows, self._flattened_length + self._current_vertices] = 0
        self._current_vertices = new_vertices
        self._state_batch[rows, self._flattened_length + self._current_vertices] = 1

        self._step_count += 1
        if self._step_count >= self.episode_length:
            self._status = EpisodeStatus.TRUNCATED

    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> Graph:
        result = (
            state_batch[:, : self._flattened_length].reshape(-1, 1, self._flattened_length).copy()
        )

        return Graph.from_flattened(
            flattened=result,
            flattened_ordering=self._flattened_ordering,
            color_representation=ColorRepresentation.BINARY_SLICES,
            is_directed=self._is_directed,
            allow_loops=self._allow_loops,
        )
