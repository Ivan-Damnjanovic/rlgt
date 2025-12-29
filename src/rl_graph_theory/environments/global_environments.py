"""
This ``Python`` module contains two global reinforcement learning environments, which inherit from
the `GraphEnvironment` class and model graph-building games where the edges (resp. arcs) are all
initially fully colored in some predetermined manner, and then in each step, any edge (resp. arc)
can be properly recolored with any color.
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
    This class inherits from the `GraphEnvironment` class and models a graph-building game in which
    the edges (resp. arcs) are all initially fully colored in some manner, and then in each step,
    any edge (resp. arc) can be properly recolored with any color. The user can select the graph
    order and the number of proper edge colors, as well as choose whether the graphs should be
    directed or undirected, and whether loops should be allowed. Additionally, the user can
    configure the mechanism that controls how the initial fully colored graphs are colored, which
    could be either deterministic or nondeterministic.

    The RL tasks in this environment are continuing, and the total number of actions to be
    performed, i.e., the episode length, is given as a configurable parameter.

    Each state is represented by a binary `numpy.ndarray` list of type `numpy.uint8` and length
    ``(edge_colors - 1) * flattened_length``, where ``edge_colors`` is the configured number of
    proper edge colors, while ``flattened_length`` is the length of the `numpy.ndarray` list from
    any of the two flattened graph formats of the graphs to be constructed. In the state vector, in
    the first ``flattened_length`` bits, the value 1 indicates which of the ``flattened_length``
    edges (resp. arcs) are currently of the color 1; in the second ``flattened_length`` bits, the
    value 1 indicates which of the ``flattened_length`` edges (resp. arcs) are currently of the
    color 2; and so on until the ``(edge_colors - 1)``-th ``flattened_length`` bits, where the
    value 1 indicates which of the ``flattened_length`` edges (resp. arcs) are currently of the
    color ``edge_colors - 1``. Here, the edges (resp. arcs) are considered to be ordered in the
    flattened row-major order or the flattened clockwise order, as explained in the `GraphFormat`
    enumeration. The user can select which of these two orderings should be applied.

    Each action is represented by a `numpy.ndarray` list of type `numpy.int32` and length two.
    Here, the first entry signifies the index of the edge (resp. arc) that should be properly
    recolored, while the second entry is a value between 0 and ``edge_colors - 1`` that determines
    which color the chosen edge (resp. arc) should be properly recolored with.

    :ivar _state_batch: See the description of the `GraphEnvironment._state_batch` attribute.
    :ivar _status: See the description of the `GraphEnvironment._status` attribute.
    :ivar _edge_colors: The number of proper edge colors in the graphs to be constructed.
    :ivar _is_directed: A boolean that indicates whether the graphs to be constructed are a
        $k$-edge-colored looped complete directed graph or a $k$-edge-colored looped complete
        undirected graph.
    :ivar _allow_loops: A boolean that indicates whether the graphs to be constructed are allowed
        to have loops.
    :ivar _flattened_ordering: An item of the `FlattenedOrdering` enumeration that determines
        whether the edges (resp. arcs) should be arranged in the flattened row-major order or the
        flattened clockwise order.
    :ivar initial_graph_generator: A `GraphGenerator` function that describes the mechanism for
        generating the underlying fully colored graphs for the states in the batch of initial
        states. It is possible to re-configure this attribute between two independent batches of
        episodes run in parallel.
    :ivar _flattened_length: A positive integer that determines the length of the `numpy.ndarray`
        list from any of the two flattened graph formats of the graphs to be constructed.
    :ivar _state_length: A positive integer that determines the length of each of the state
        vectors, i.e., the number ``(_edge_colors - 1) * _flattened_length``.
    :ivar episode_length: A positive integer that determines the episode length of each of the
        episodes. It is possible to re-configure this attribute between two independent batches of
        episodes run in parallel.
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
    ):
        """
        This constructor initializes an instance of the `GlobalSetEnvironment` object.

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
        """

        super().__init__(reward_type=reward_type, reward_function=reward_function)

        self._edge_colors: int = edge_colors
        self._is_directed: bool = is_directed
        self._allow_loops: bool = allow_loops
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

        self._state_length: int = (self._edge_colors - 1) * self._flattened_length

        if episode_length is not None:
            self.episode_length: int = episode_length
        # If the ``episode_length`` argument is `None`, then the episode length should match the
        # ``_flattened_length`` attribute.
        else:
            self.episode_length: int = self._flattened_length

        self._step_count: Optional[int] = None

    def reset_batch(self, batch_size: int) -> Tuple[np.ndarray, EpisodeStatus]:
        # Use the ``initial_graph_generator`` to generate the initial underlying fully colored
        # graphs.
        initial_graph_batch = self.initial_graph_generator(batch_size=batch_size)
        if self._flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            format_representation = initial_graph_batch.flattened_row_major
        else:
            format_representation = initial_graph_batch.flattened_clockwise

        # Initialize the state vectors using the generated underlying fully colored graphs.
        if self._edge_colors == 2:
            self._state_batch = format_representation.copy()
        else:
            color_indices = np.arange(1, self._edge_colors, dtype=np.uint8)
            temp = (format_representation[:, None, :] == color_indices[:, None]).astype(np.uint8)
            self._state_batch = temp.reshape(-1, self._state_length)

        self._status = EpisodeStatus.IN_PROGRESS
        self._step_count = 0

        return self._state_batch, self._status

    def _transition_batch(self, action_batch: np.ndarray) -> None:
        """
        This method performs the transition process by executing a batch of given actions and
        applying them element-wise to the states in the batch of current states given by the
        `_state_batch` attribute.

        :param action_batch: The batch of actions to be applied to the states in the batch of
            current states, given as a `numpy.ndarray` matrix of type `numpy.int32` where the rows
            correspond to the actions. The number of actions in this batch must be the same as the
            number of states in the `_state_batch` attribute.
        """

        rows = np.arange(self._state_batch.shape[0], dtype=np.int32)

        # If the graphs have only two proper edge colors, then the transition can easily be done as
        # follows.
        if self._edge_colors == 2:
            self._state_batch[rows, action_batch[:, 0]] = action_batch[:, 1]
        # Otherwise, the next trick should be used.
        else:
            temp = self._state_batch.reshape(-1, self._edge_colors - 1, self._flattened_length)
            temp[rows, :, action_batch[:, 0]] = 0
            temp[rows, action_batch[:, 1] - 1, action_batch[:, 0]] = action_batch[:, 1] != 0

        self._step_count += 1
        if self._step_count >= self.episode_length:
            self._status = EpisodeStatus.TRUNCATED

    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> GraphBatch:
        # If the graphs have only two proper edge colors, then the conversion follows immediately.
        if self._edge_colors == 2:
            return GraphBatch.from_flattened(
                flattened=state_batch,
                flattened_ordering=self._flattened_ordering,
                is_directed=self._is_directed,
                allow_loops=self._allow_loops,
            )

        # Otherwise, we reconstruct the flattened format for the underlying graphs from the states.
        temp = state_batch.reshape(-1, self._edge_colors - 1, self._flattened_length)
        color_indices = np.arange(1, self._edge_colors, dtype=np.uint8)
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
    This class inherits from the `GraphEnvironment` class and models a graph-building game used for
    constructing 2-edge-colored looped complete graphs where the edges (resp. arcs) are all
    initially fully colored in some manner, and then in each step, any edge (resp. arc) can be
    potentially flipped. The user can select the graph order, as well as choose whether the graphs
    should be directed or undirected, and whether loops should be allowed. Additionally, the user
    can configure the mechanism that controls how the initial fully colored graphs are colored,
    which could be either deterministic or nondeterministic.

    The RL tasks in this environment are continuing, and the total number of actions to be
    performed, i.e., the episode length, is given as a configurable parameter.

    Each state is represented by a binary `numpy.ndarray` list of type `numpy.uint8` that describes
    the current graph in one of the two flattened graph formats. The user can select which of these
    two formats should be applied throughout the entire construction procedure, i.e., how the edges
    (resp. arcs) should be arranged, and this parameter cannot be re-configured.

    The RL environment has two modes of operations that affect how the actions are handled. If the
    ``flip_only`` parameter is set to `False`, which is the default value, then each action is
    represented by a `numpy.ndarray` list of type `numpy.int32` and length two. In this case, the
    first entry signifies the index of the edge (resp. arc) that should be potentially flipped,
    while the second entry is a binary value that determines whether the selected edge (resp. arc)
    should be flipped. In other words, the number 1 indicates that the proper edge color of the
    selected edge (resp. arc) should be changed, while the number 0 indicates that the proper edge
    color of the selected edge (resp. arc) should stay the same. On the other hand, if the
    ``flip_only`` parameter is set to `True`, then any selected edge (resp. arc) must be flipped.
    In this case, each action is represented by a single-entry `numpy.ndarray` list of type
    `numpy.int32` containing the index of the edge (resp. arc) that should be flipped, i.e., whose
    proper edge color should be changed.

    :ivar _state_batch: See the description of the `GraphEnvironment._state_batch` attribute.
    :ivar _status: See the description of the `GraphEnvironment._status` attribute.
    :ivar _is_directed: A boolean that indicates whether the graphs to be constructed are a
        $k$-edge-colored looped complete directed graph or a $k$-edge-colored looped complete
        undirected graph.
    :ivar _allow_loops: A boolean that indicates whether the graphs to be constructed are allowed
        to have loops.
    :ivar _flip_only: A boolean that indicates whether all the selected edges (resp. arcs) should
        only be allowed to be flipped, i.e., the value of the ``flip_only`` parameter.
    :ivar _flattened_ordering: An item of the `FlattenedOrdering` enumeration that determines
        whether the edges (resp. arcs) should be arranged in the flattened row-major order or the
        flattened clockwise order.
    :ivar initial_graph_generator: A `GraphGenerator` function that describes the mechanism for
        generating the underlying fully colored graphs for the states in the batch of initial
        states. It is possible to re-configure this attribute between two independent batches of
        episodes run in parallel.
    :ivar _flattened_length: A positive integer that determines the length of the `numpy.ndarray`
        list from any of the two flattened graph formats of the graphs to be constructed.
    :ivar episode_length: A positive integer that determines the episode length of each of the
        episodes. It is possible to re-configure this attribute between two independent batches of
        episodes run in parallel.
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
    ):
        """
        This constructor initializes an instance of the `GlobalFlipEnvironment` object.

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
        :param flip_only: A boolean that indicates whether all the selected edges (resp. arcs)
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
        """

        super().__init__(reward_type=reward_type, reward_function=reward_function)

        self._is_directed: bool = is_directed
        self._allow_loops: bool = allow_loops
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

        self._step_count: Optional[int] = None

    def reset_batch(self, batch_size: int) -> Tuple[np.ndarray, EpisodeStatus]:
        # Use the ``initial_graph_generator`` to generate the initial underlying fully colored
        # graphs.
        initial_graph_batch = self.initial_graph_generator(batch_size=batch_size)
        if self._flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            format_representation = initial_graph_batch.flattened_row_major
        else:
            format_representation = initial_graph_batch.flattened_clockwise

        # Initialize the state vectors using the generated underlying fully colored graphs.
        self._state_batch = format_representation.copy()
        self._status = EpisodeStatus.IN_PROGRESS
        self._step_count = 0

        return self._state_batch, self._status

    def _transition_batch(self, action_batch: np.ndarray) -> None:
        """
        This method performs the transition process by executing a batch of given actions and
        applying them element-wise to the states in the batch of current states given by the
        `_state_batch` attribute.

        :param action_batch: The batch of actions to be applied to the states in the batch of
            current states, given as a `numpy.ndarray` matrix of type `numpy.int32` where the rows
            correspond to the actions. The number of actions in this batch must be the same as the
            number of states in the `_state_batch` attribute.
        """

        rows = np.arange(self._state_batch.shape[0], dtype=np.int32)
        # If the ``_flip_only`` attribute is `True`, then all the selected edges (resp. arcs) must
        # be flipped.
        if self._flip_only:
            self._state_batch[rows, action_batch[:, 0]] ^= 1
        # Otherwise, each selected edge (resp. arc) is either flipped or not flipped, depending on
        # the second entry from the corresponding action, which is binary.
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
