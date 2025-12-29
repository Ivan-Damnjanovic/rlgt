"""
This ``Python`` module contains two local reinforcement learning environments, which inherit from
the `GraphEnvironment` class and model graph-building games where the edges (resp. arcs) are all
initially fully colored in some predetermined manner, and the agent moves from one vertex to
another in some way, thereby traversing the existing edges (resp. arcs) and properly recoloring
them.
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


def __compute_edge_indices(
    graph_order: int,
    starting_vertices: np.ndarray,
    ending_vertices: np.ndarray,
    flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
    is_directed: bool = False,
    allow_loops: bool = False,
) -> np.ndarray:
    """
    This auxiliary function considers a $k$-edge-colored looped complete graph, and computes the
    index of each of the edges (resp. arcs) from a given list, with respect to the flattened
    row-major format or the flattened clockwise format, as described in the `GraphFormat`
    enumeration. The edges (resp. arcs) are given as ordered pairs of vertices consisting of the
    starting vertex and the ending vertex. It is possible to configure which of the two flattened
    orderings should be used to arrange the edges (resp. arcs), as well as select the graph order
    and choose whether the graphs should be directed or undirected, and whether loops should be
    allowed. In the case of undirected graphs, it does not matter which vertex is the starting one
    and which is the ending. If loops are not allowed, then the starting and the ending vertex from
    each pair must be distinct.

    :param graph_order: A positive integer that determines the order of the considered graph.
    :param starting_vertices: A `numpy.ndarray` list of type `numpy.int32` that contains the first
        vertex from each of the given ordered pairs of vertices that represent the edges (resp.
        arcs).
    :param ending_vertices: A `numpy.ndarray` list of type `numpy.int32` that contains the second
        vertex from each of the given ordered pairs of vertices that represent the edges (resp.
        arcs).
    :param flattened_ordering: An item of the `FlattenedOrdering` enumeration that determines
        whether the edges (resp. arcs) should be arranged in the flattened row-major order or the
        flattened clockwise order. The default value is `FlattenedOrdering.ROW_MAJOR`, i.e., the
        edges (resp. arcs) should be arranged in the flattened row-major order by default.
    :param is_directed: A boolean that indicates whether the considered graph is a $k$-edge-colored
        looped complete directed graph or a $k$-edge-colored looped complete undirected graph. The
        default value is `False`, i.e., the considered graph is undirected by default.
    :param allow_loops: A boolean that indicates whether the considered graph is allowed to have
        loops. The default value is `False`, i.e., the considered graph is not allowed to have
        loops by default.
    """

    if is_directed:
        # Settle the case for the directed graphs with the flattened row-major order.
        if flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            if allow_loops:
                result = starting_vertices * graph_order + ending_vertices
            else:
                result = (
                    starting_vertices * (graph_order - 1)
                    + ending_vertices
                    - (ending_vertices >= starting_vertices).astype(np.int32)
                )
        # Settle the case for the directed graphs with the flattened clockwise order.
        else:
            layer = np.maximum(starting_vertices, ending_vertices)

            if allow_loops:
                result = layer * layer + layer - ending_vertices + starting_vertices
            else:
                result = (
                    layer * layer
                    - ending_vertices
                    + starting_vertices
                    - (ending_vertices <= starting_vertices).astype(np.int32)
                )

    # Settle the case for the undirected graphs.
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
    This class inherits from the `GraphEnvironment` class and models a graph-building game where
    the edges (resp. arcs) are all initially fully colored in some manner, and the agent moves from
    one vertex to another in some way, thereby traversing the existing edges (resp. arcs) and
    properly recoloring them. More precisely, in each step, the agent is located at some vertex and
    needs to select an edge incident to this vertex (resp. an arc starting at this vertex), then
    traverse it and move to the other endpoint of the traversed edge (resp. arc). While traversing
    an edge (resp. arc), the agent also properly recolors it with a selected color. The user can
    select the graph order and the number of proper edge colors, as well as choose whether the
    graphs should be directed or undirected, and whether loops should be allowed. Additionally, the
    user can configure the mechanism that controls how the initial fully colored graphs are
    colored, which could be either deterministic or nondeterministic. The user can also select the
    vertex at which the agent should start the recoloring procedure.

    The RL tasks in this environment are continuing, and the total number of actions to be
    performed, i.e., the episode length, is given as a configurable parameter.

    Each state is represented by a binary `numpy.ndarray` list of type `numpy.uint8` and length
    ``(edge_colors - 1) * flattened_length + graph_order``, where ``edge_colors`` is the configured
    number of proper edge colors and ``graph_order`` is the configured graph order, while
    ``flattened_length`` is the length of the `numpy.ndarray` list from any of the two flattened
    graph formats of the graphs to be constructed. In the state vector, in the first
    ``flattened_length`` bits, the value 1 indicates which of the ``flattened_length`` edges (resp.
    arcs) are currently of the color 1; in the second ``flattened_length`` bits, the value 1
    indicates which of the ``flattened_length`` edges (resp. arcs) are currently of the color 2;
    and so on until the ``(edge_colors - 1)``-th ``flattened_length`` bits, where the value 1
    indicates which of the ``flattened_length`` edges (resp. arcs) are currently of the color
    ``edge_colors - 1``. Here, the edges (resp. arcs) are considered to be ordered in the flattened
    row-major order or the flattened clockwise order, as explained in the `GraphFormat`
    enumeration. The user can select which of these two orderings should be applied. The final
    ``graph_order`` bits from the state vector represent a one-hot encoding of the position that
    determines the vertex where the agent is currently located. In other words, there is exactly
    one value of 1 whose index determines the vertex where the agent is currently located.

    Each action is represented by a `numpy.ndarray` list of type `numpy.int8` and length two. Here,
    the first entry signifies the vertex that the agent should move to from the current vertex,
    while the second entry is a value between 0 and ``edge_colors - 1`` that determines which color
    the traversed edge (resp. arc) should be recolored with. If loops are not allowed, then the
    first entry of the action vector must be distinct from the vertex where the agent is currently
    located.

    :ivar _state_batch: See the description of the `GraphEnvironment._state_batch` attribute.
    :ivar _status: See the description of the `GraphEnvironment._status` attribute.
    :ivar _edge_colors: The number of proper edge colors in the graphs to be constructed.
    :ivar _is_directed: A boolean that indicates whether the graphs to be constructed are a
        $k$-edge-colored looped complete directed graph or a $k$-edge-colored looped complete
        undirected graph.
    :ivar _allow_loops: A boolean that indicates whether the graphs to be constructed are allowed
        to have loops.
    :ivar _graph_order: A positive integer that describes the order of the graphs to be
        constructed.
    :ivar _flattened_ordering: An item of the `FlattenedOrdering` enumeration that determines
        whether the edges (resp. arcs) should be arranged in the flattened row-major order or the
        flattened clockwise order.
    :ivar initial_graph_generator: A `GraphGenerator` function that describes the mechanism for
        generating the underlying fully colored graphs for the states in the batch of initial
        states. It is possible to re-configure this attribute between two independent batches of
        episodes run in parallel.
    :ivar starting_vertex: A nonnegative integer below the configured graph order that determines
        the vertex at which the agent should start the recoloring procedure. It is possible to
        re-configure this attribute between two independent batches of episodes run in parallel.
    :ivar _flattened_length: A positive integer that determines the length of the `numpy.ndarray`
        list from any of the two flattened graph formats of the graphs to be constructed.
    :ivar _state_length: A positive integer that determines the length of each of the state
        vectors, i.e., the number ``(_edge_colors - 1) * _flattened_length + _graph_order``.
    :ivar episode_length: A positive integer that determines the episode length of each of the
        episodes. It is possible to re-configure this attribute between two independent batches of
        episodes run in parallel.
    :ivar _current_vertices: Either `None`, or a `numpy.ndarray` list of type `numpy.int8` that
        determines the vertex where the agent is currently located in each of the episodes run in
        parallel, in the natural order. This attribute is initially set to `None`, and afterwards,
        it is assigned the necessary `numpy.ndarray` list after each invocation of the
        `reset_batch` or `GraphEnvironment.step_batch` method.
    :ivar _step_count: Either `None`, or a nonnegative integer that signifies how many steps have
        been taken, i.e., how many times an action has been executed, in each of the episodes from
        the current batch. When this number becomes `episode_length`, this indicates that a final
        state has been reached and that the episode has been truncated. This attribute is initially
        set to `None`, and afterwards, it is assigned the necessary `int` value after each
        invocation of the `reset_batch` or `GraphEnvironment.step_batch` method.
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
        starting_vertex: int = 0,
    ):
        """
        This constructor initializes an instance of the `LocalSetEnvironment` object.

        :param reward_type: An item of the `RewardType` enumeration that determines the (sub)type
            of reward system to be used in the instantiated environment.
        :param reward_function: The `RewardFunction` function whose goal is to help compute the
            rewards in accordance with the selected (sub)type of reward system. It plays the role
            of either the ``graph_invariant`` or the ``graph_invariant_difference`` function from
            the description of the `RewardType` enumeration, and its expected signature varies
            depending on the ``reward_type`` argument.
        :param graph_order: A positive integer (not below two) that represents the graph order of
            the graphs to be constructed.
        :param episode_length: Either `None`, or a positive integer that determines the episode
            length of each of the episodes. If the argument is `None`, then this means that the
            episode length should match the number of entries in any of the two flattened graph
            formats of the graphs to be constructed. The default value is `None`.
        :param flattened_ordering: An item of the `FlattenedOrdering` enumeration that determines
            whether the edges (resp. arcs) should be arranged in the flattened row-major order or
            the flattened clockwise order. The default value is `FlattenedOrdering.ROW_MAJOR`,
            i.e., the edges (resp. arcs) should be arranged in the flattened row-major order by
            default.
        :param edge_colors: A positive integer (not below two) that represents the number of proper
            edge colors in the graphs to be constructed. The default value is two.
        :param is_directed: A boolean that indicates whether the graphs to be constructed are a
            $k$-edge-colored looped complete directed graph or a $k$-edge-colored looped complete
            undirected graph. The default value is `False`, i.e., the graphs to be constructed are
            undirected by default.
        :param allow_loops: A boolean that indicates whether the graphs to be constructed are
            allowed to have loops. The default value is `False`, i.e., the graphs to be constructed
            are not allowed to have loops by default.
        :param initial_graph_generator: Either `None`, or a `GraphGenerator` function that
            describes the mechanism for generating the underlying fully colored graphs for the
            states in the batch of initial states. If the argument is `None`, then this means that
            all the edges (resp. arcs) from all the graphs should initially be colored with the
            color 0. The default value is `None`.
        :param starting_vertex: A nonnegative integer between 0 and ``graph_order - 1`` that
            determines the vertex at which the agent should start the recoloring procedure. The
            default value is 0.
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
            # By default, all the edges (resp. arcs) from all the graphs should be colored with the
            # color 0.
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

        self.starting_vertex: int = starting_vertex

        # Compute the number of entries in any of the two flattened graph formats depending on the
        # selected graph order, on whether the constructed graphs should be directed, and on
        # whether loops should be allowed.
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

        self._state_length: int = (
            self._edge_colors - 1
        ) * self._flattened_length + self._graph_order

        if episode_length is not None:
            self.episode_length: int = episode_length
        # If the ``episode_length`` argument is `None`, then the episode length should match the
        # ``_flattened_length`` attribute.
        else:
            self.episode_length: int = self._flattened_length

        self._current_vertices: Optional[np.ndarray] = None
        self._step_count: Optional[int] = None

    def reset_batch(self, batch_size: int) -> Tuple[np.ndarray, EpisodeStatus]:
        # Use the ``initial_graph_generator`` to generate the initial underlying fully colored
        # graphs.
        initial_graph_batch = self.initial_graph_generator(batch_size=batch_size)
        if self._flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            format_representation = initial_graph_batch.flattened_row_major
        else:
            format_representation = initial_graph_batch.flattened_clockwise

        # Make sure that the current vertex position flags are all set to the configured starting
        # vertex.
        self._state_batch = np.zeros((batch_size, self._state_length), dtype=np.uint8)
        self._state_batch[:, -self._graph_order + self.starting_vertex] = 1

        # Finish initializing the state vectors using the generated underlying fully colored
        # graphs.
        if self._edge_colors == 2:
            self._state_batch[:, : self._flattened_length] = format_representation
        else:
            color_indices = np.arange(1, self._edge_colors, dtype=np.uint8)
            temp = (format_representation[:, None, :] == color_indices[:, None]).astype(np.uint8)
            self._state_batch[:, : -self._graph_order] = temp.reshape(temp.shape[0], -1)

        self._current_vertices = np.full((batch_size,), self.starting_vertex, dtype=np.int8)
        self._status = EpisodeStatus.IN_PROGRESS
        self._step_count = 0

        return self._state_batch, self._status

    def _transition_batch(self, action_batch: np.ndarray) -> None:
        """
        This method performs the transition process by executing a batch of given actions and
        applying them element-wise to the states in the batch of current states given by the
        `_state_batch` attribute.

        :param action_batch: The batch of actions to be applied to the states in the batch of
            current states, given as a `numpy.ndarray` matrix of type `numpy.int8` where the rows
            correspond to the actions. The number of actions in this batch must be the same as the
            number of states in the `_state_batch` attribute.

        :note: If loops are not allowed and the agent attempts to traverse a loop, a `RuntimeError`
            will be raised.
        """

        # If the agent attempts to traverse a nonexisting loop, a `RuntimeError` will be raised.
        if not self._allow_loops:
            if np.any(self._current_vertices == action_batch[:, 0]):
                raise RuntimeError

        # Use the ``__compute_edge_indices`` auxiliary function to compute the index of the edge
        # (resp. arc) that should be traversed in each of the episodes run in parallel.
        edge_indices = __compute_edge_indices(
            graph_order=self._graph_order,
            starting_vertices=self._current_vertices,
            ending_vertices=action_batch[:, 0],
            flattened_ordering=self._flattened_ordering,
            is_directed=self._is_directed,
            allow_loops=self._allow_loops,
        )

        rows = np.arange(self._state_batch.shape[0], dtype=np.int32)

        # If the graphs have only two proper edge colors, then the transition can easily be done as
        # follows.
        if self._edge_colors == 2:
            self._state_batch[rows, edge_indices] = action_batch[:, 1]
        # Otherwise, the next trick should be used.
        else:
            temp = self._state_batch[:, : -self._graph_order].reshape(
                -1, self._edge_colors - 1, self._flattened_length
            )
            temp[rows, :, edge_indices] = 0
            temp[rows, action_batch[:, 1] - 1, edge_indices] = action_batch[:, 1] != 0
            self._state_batch[:, : -self._graph_order] = temp.reshape(temp.shape[0], -1)

        # Update the current vertex position flags.
        self._state_batch[rows, -self._graph_order + self._current_vertices] = 0
        self._current_vertices = action_batch[:, 0]
        self._state_batch[rows, -self._graph_order + self._current_vertices] = 1

        self._step_count += 1
        if self._step_count >= self.episode_length:
            self._status = EpisodeStatus.TRUNCATED

    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> GraphBatch:
        # If the graphs have only two proper edge colors, then the conversion follows immediately.
        if self._edge_colors == 2:
            return GraphBatch.from_flattened(
                flattened=state_batch[:, : self._flattened_length],
                flattened_ordering=self._flattened_ordering,
                is_directed=self._is_directed,
                allow_loops=self._allow_loops,
            )

        # Otherwise, we reconstruct the flattened format for the underlying graphs from the states.
        temp = state_batch[:, : -self._graph_order].reshape(
            -1, self._edge_colors - 1, self._flattened_length
        )
        color_indices = np.arange(1, self._edge_colors, dtype=np.uint8)
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
    This class inherits from the `GraphEnvironment` class and models a graph-building game used for
    constructing 2-edge-colored looped complete graphs where the edges (resp. arcs) are all
    initially fully colored in some manner, and the agent moves from one vertex to another in some
    way, thereby traversing the existing edges (resp. arcs) and potentially flipping them. More
    precisely, in each step, the agent is located at some vertex and needs to select an edge
    incident to this vertex (resp. an arc starting at this vertex), then traverse it and move to
    the other endpoint of the traversed edge (resp. arc). While traversing an edge (resp. arc), the
    agent also potentially flip it. The user can select the graph order, as well as choose whether
    the graphs should be directed or undirected, and whether loops should be allowed. Additionally,
    the user can configure the mechanism that controls how the initial fully colored graphs are
    colored, which could be either deterministic or nondeterministic. The user can also select the
    vertex at which the agent should start the potential flipping procedure.

    The RL tasks in this environment are continuing, and the total number of actions to be
    performed, i.e., the episode length, is given as a configurable parameter.

    Each state is represented by a binary `numpy.ndarray` list of type `numpy.uint8` and length
    ``flattened_length + graph_order``, where ``graph_order`` is the configured graph order, while
    ``flattened_length`` is the length of the `numpy.ndarray` list from any of the two flattened
    graph formats of the graphs to be constructed. In the state vector, in the first
    ``flattened_length`` bits, the value 1 indicates which of the ``flattened_length`` edges (resp.
    arcs) are currently of the color 1. Here, the edges (resp. arcs) are considered to be ordered
    in the flattened row-major order or the flattened clockwise order, as explained in the
    `GraphFormat` enumeration. The user can select which of these two orderings should be applied.
    The final ``graph_order`` bits from the state vector represent a one-hot encoding of the
    position that determines the vertex where the agent is currently located. In other words, there
    is exactly one value of 1 whose index determines the vertex where the agent is currently
    located.

    The RL environment has two modes of operations that affect how the actions are handled. If the
    ``flip_only`` parameter is set to `False`, which is the default value, then each action is
    represented by a `numpy.ndarray` list of type `numpy.int8` and length two. In this case, the
    first entry signifies the vertex that the agent should move to from the current vertex, while
    the second entry is a binary value that determines whether the traversed edge (resp. arc)
    should be flipped. In other words, the number 1 indicates that the proper edge color of the
    traversed edge (resp. arc) should be changed, while the number 0 indicates that the proper edge
    color of the traversed edge (resp. arc) should stay the same. On the other hand, if the
    ``flip_only`` parameter is set to `True`, then any traversed edge (resp. arc) must be flipped.
    In this case, each action is represented by a single-entry `numpy.ndarray` list of type
    `numpy.int8` containing the vertex that the agent should move to from the current vertex, with
    the traversed edge (resp. arc) being necessarily flipped. If loops are not allowed, then the
    first (and potentially only) entry of the action vector must be distinct from the vertex where
    the agent is currently located.

    :ivar _state_batch: See the description of the `GraphEnvironment._state_batch` attribute.
    :ivar _status: See the description of the `GraphEnvironment._status` attribute.
    :ivar _is_directed: A boolean that indicates whether the graphs to be constructed are a
        2-edge-colored looped complete directed graph or a 2-edge-colored looped complete
        undirected graph.
    :ivar _allow_loops: A boolean that indicates whether the graphs to be constructed are allowed
        to have loops.
    :ivar _graph_order: A positive integer that describes the order of the graphs to be
        constructed.
    :ivar _flip_only: A boolean that indicates whether all the traversed edges (resp. arcs) should
        only be allowed to be flipped, i.e., the value of the ``flip_only`` parameter.
    :ivar _flattened_ordering: An item of the `FlattenedOrdering` enumeration that determines
        whether the edges (resp. arcs) should be arranged in the flattened row-major order or the
        flattened clockwise order.
    :ivar initial_graph_generator: A `GraphGenerator` function that describes the mechanism for
        generating the underlying fully colored graphs for the states in the batch of initial
        states. It is possible to re-configure this attribute between two independent batches of
        episodes run in parallel.
    :ivar starting_vertex: A nonnegative integer below the configured graph order that determines
        the vertex at which the agent should start the (potential) flipping procedure. It is
        possible to re-configure this attribute between two independent batches of episodes run in
        parallel.
    :ivar _flattened_length: A positive integer that determines the length of the `numpy.ndarray`
        list from any of the two flattened graph formats of the graphs to be constructed.
    :ivar episode_length: A positive integer that determines the episode length of each of the
        episodes. It is possible to re-configure this attribute between two independent batches of
        episodes run in parallel.
    :ivar _current_vertices: Either `None`, or a `numpy.ndarray` list of type `numpy.int8` that
        determines the vertex where the agent is currently located in each of the episodes run in
        parallel, in the natural order. This attribute is initially set to `None`, and afterwards,
        it is assigned the necessary `numpy.ndarray` list after each invocation of the
        `reset_batch` or `GraphEnvironment.step_batch` method.
    :ivar _step_count: Either `None`, or a nonnegative integer that signifies how many steps have
        been taken, i.e., how many times an action has been executed, in each of the episodes from
        the current batch. When this number becomes `episode_length`, this indicates that a final
        state has been reached and that the episode has been truncated. This attribute is initially
        set to `None`, and afterwards, it is assigned the necessary `int` value after each
        invocation of the `reset_batch` or `GraphEnvironment.step_batch` method.
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
        starting_vertex: int = 0,
    ):
        """
        This constructor initializes an instance of the `LocalFlipEnvironment` object.

        :param reward_type: An item of the `RewardType` enumeration that determines the (sub)type
            of reward system to be used in the instantiated environment.
        :param reward_function: The `RewardFunction` function whose goal is to help compute the
            rewards in accordance with the selected (sub)type of reward system. It plays the role
            of either the ``graph_invariant`` or the ``graph_invariant_difference`` function from
            the description of the `RewardType` enumeration, and its expected signature varies
            depending on the ``reward_type`` argument.
        :param graph_order: A positive integer (not below two) that represents the graph order of
            the graphs to be constructed.
        :param episode_length: Either `None`, or a positive integer that determines the episode
            length of each of the episodes. If the argument is `None`, then this means that the
            episode length should match the number of entries in any of the two flattened graph
            formats of the graphs to be constructed. The default value is `None`.
        :param flip_only: A boolean that indicates whether all the traversed edges (resp. arcs)
            should only be allowed to be flipped. The default value is `False`.
        :param flattened_ordering: An item of the `FlattenedOrdering` enumeration that determines
            whether the edges (resp. arcs) should be arranged in the flattened row-major order or
            the flattened clockwise order. The default value is `FlattenedOrdering.ROW_MAJOR`,
            i.e., the edges (resp. arcs) should be arranged in the flattened row-major order by
            default.
        :param is_directed: A boolean that indicates whether the graphs to be constructed are a
            2-edge-colored looped complete directed graph or a 2-edge-colored looped complete
            undirected graph. The default value is `False`, i.e., the graphs to be constructed are
            undirected by default.
        :param allow_loops: A boolean that indicates whether the graphs to be constructed are
            allowed to have loops. The default value is `False`, i.e., the graphs to be constructed
            are not allowed to have loops by default.
        :param initial_graph_generator: Either `None`, or a `GraphGenerator` function that
            describes the mechanism for generating the underlying fully colored graphs for the
            states in the batch of initial states. If the argument is `None`, then this means that
            all the edges (resp. arcs) from all the graphs should initially be colored with the
            color 0. The default value is `None`.
        :param starting_vertex: A nonnegative integer between 0 and ``graph_order - 1`` that
            determines the vertex at which the agent should start the (potential) flipping
            procedure. The default value is 0.
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
            # By default, all the edges (resp. arcs) from all the graphs should be colored with the
            # color 0.
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

        self.starting_vertex: int = starting_vertex

        # Compute the number of entries in any of the two flattened graph formats depending on the
        # selected graph order, on whether the constructed graphs should be directed, and on
        # whether loops should be allowed.
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

        if episode_length is not None:
            self.episode_length: int = episode_length
        # If the ``episode_length`` argument is `None`, then the episode length should match the
        # ``_flattened_length`` attribute.
        else:
            self.episode_length: int = self._flattened_length

        self._current_vertices: Optional[np.ndarray] = None
        self._step_count: Optional[int] = None

    def reset_batch(self, batch_size: int) -> Tuple[np.ndarray, EpisodeStatus]:
        # Use the ``initial_graph_generator`` to generate the initial underlying fully colored
        # graphs.
        initial_graph_batch = self.initial_graph_generator(batch_size=batch_size)
        if self._flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            format_representation = initial_graph_batch.flattened_row_major
        else:
            format_representation = initial_graph_batch.flattened_clockwise

        # Initialize the state vectors using the generated underlying fully colored graphs. Also,
        # make sure that the current vertex position flags are all set to the configured starting
        # vertex.
        self._state_batch = np.zeros(
            (batch_size, self._flattened_length + self._graph_order), dtype=np.uint8
        )
        self._state_batch[:, : self._flattened_length] = format_representation
        self._state_batch[:, self._flattened_length + self.starting_vertex] = 1

        self._current_vertices = np.full((batch_size,), self.starting_vertex, dtype=np.int8)
        self._status = EpisodeStatus.IN_PROGRESS
        self._step_count = 0

        return self._state_batch, self._status

    def _transition_batch(self, action_batch: np.ndarray) -> None:
        """
        This method performs the transition process by executing a batch of given actions and
        applying them element-wise to the states in the batch of current states given by the
        `_state_batch` attribute.

        :param action_batch: The batch of actions to be applied to the states in the batch of
            current states, given as a `numpy.ndarray` matrix of type `numpy.int8` where the rows
            correspond to the actions. The number of actions in this batch must be the same as the
            number of states in the `_state_batch` attribute.

        :note: If loops are not allowed and the agent attempts to traverse a loop, a `RuntimeError`
            will be raised.
        """

        # If the agent attempts to traverse a nonexisting loop, a `RuntimeError` will be raised.
        if not self._allow_loops:
            if np.any(self._current_vertices == action_batch[:, 0]):
                raise RuntimeError

        # Use the ``__compute_edge_indices`` auxiliary function to compute the index of the edge
        # (resp. arc) that should be traversed in each of the episodes run in parallel.
        edge_indices = __compute_edge_indices(
            graph_order=self._graph_order,
            starting_vertices=self._current_vertices,
            ending_vertices=action_batch[:, 0],
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
            self._state_batch[rows, edge_indices] ^= action_batch[:, 1]

        # Update the current vertex position flags.
        self._state_batch[rows, self._flattened_length + self._current_vertices] = 0
        self._current_vertices = action_batch[:, 0]
        self._state_batch[rows, self._flattened_length + self._current_vertices] = 1

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
