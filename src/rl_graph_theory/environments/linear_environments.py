"""
This ``Python`` module contains three linear reinforcement learning environments, which inherit
from the `GraphEnvironment` class and model graph-building games where the edges (resp. arcs) are
all initially either uncolored, or colored in some predetermined manner, and are then (re)colored
one by one, either in the flattened row-major order or the flattened clockwise order.
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


class LinearBuildEnvironment(GraphEnvironment):
    """
    This class inherits from the `GraphEnvironment` class and models a graph-building game in which
    the edges (resp. arcs) are all initially uncolored and are then colored one by one, either in
    the flattened row-major order or the flattened clockwise order. The user can select the graph
    order and the number of proper edge colors, as well as choose whether the graphs should be
    directed or undirected, and whether loops should be allowed. The edges (resp. arcs) are colored
    in the same way that a graph is reconstructed from a flattened format. In other words, if loops
    are not allowed, then these edges (resp. arcs) are just deleted and completely ignored, and if
    the graphs are undirected, then only the upper triangular part of an adjacency matrix is
    considered (with or without the diagonal, depending on whether loops are allowed) and each edge
    is essentially colored in both directions at the same time. The user can choose whether the
    edges (resp. arcs) should be colored in the flattened row-major order or the flattened
    clockwise order.

    The RL tasks in this environment are episodic, and the total number of actions to be performed
    equals the number of entries in each of the two flattened graph formats. Therefore, this number
    depends on the selected graph order, on whether the constructed graphs are directed, and on
    whether loops are allowed.

    Each state is represented by a binary `numpy.ndarray` list of type `numpy.uint8` and length
    ``edge_colors * flattened_length``, where ``edge_colors`` is the configured number of proper
    edge colors, while ``flattened_length`` is the length of the `numpy.ndarray` list from any of
    the two flattened graph formats of the graphs to be constructed. In the state vector, in the
    first ``flattened_length`` bits, the value 1 indicates which of the ``flattened_length`` edges
    (resp. arcs) has been colored with the color 1; in the second ``flattened_length`` bits, the
    value 1 indicates which of the ``flattened_length`` edges (resp. arcs) has been colored with
    the color 2; and so on until the ``(edge_colors - 1)``-th ``flattened_length`` bits, where the
    value 1 indicates which of the ``flattened_length`` edges (resp. arcs) has been colored with
    the color ``edge_colors - 1``. The final ``flattened_length`` bits represent a one-hot encoding
    of the position that determines the next edge (resp. arc) to be colored. In other words, there
    is either one value of 1 whose index determines which edge (resp. arc) should be colored next,
    or all the values are 0 and this signifies a terminal state, i.e., a state where all the edges
    (resp. arcs) have been colored.

    Each action is represented by a single-entry `numpy.ndarray` list of type `numpy.int8`
    containing a value between 0 and ``edge_colors - 1`` that determines which color the next edge
    (resp. arc) should be colored with.

    :ivar _edge_colors: The number of proper edge colors in the graphs to be constructed.
    :ivar _is_directed: A boolean that indicates whether the graphs to be constructed are a
        $k$-edge-colored looped complete directed graph or a $k$-edge-colored looped complete
        undirected graph.
    :ivar _allow_loops: A boolean that indicates whether the graphs to be constructed are allowed
        to have loops.
    :ivar _flattened_ordering: An item of the `FlattenedOrdering` enumeration that determines
        whether the edges (resp. arcs) should be colored in the flattened row-major order or the
        flattened clockwise order.
    :ivar _flattened_length: A positive integer that determines the length of the `numpy.ndarray`
        list from any of the two flattened graph formats of the graphs to be constructed. This
        number also represents how many actions are needed in total to reach a terminal state from
        an initial state.
    :ivar _state_length: A positive integer that determines the length of each of the state
        vectors, i.e., the number ``_edge_colors * _flattened_length``.
    :ivar _step_count: An integer between 0 and ``_flattened_length`` that signifies the index of
        the next edge (resp. arc) to be colored, in the order determined by the
        ``_flattened_ordering`` attribute. The value of ``_flattened_length`` indicates that all
        the edges (resp. arcs) have been colored and that no edge (resp. arc) should be colored
        next, i.e., a terminal state has been reached.
    """

    def __init__(
        self,
        reward_type: RewardType,
        reward_function: RewardFunction,
        graph_order: int,
        flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
        edge_colors: int = 2,
        is_directed: bool = False,
        allow_loops: bool = False,
    ):
        """
        This constructor initializes an instance of the `LinearBuildEnvironment` object.

        :param reward_type: An item of the `RewardType` enumeration that determines the (sub)type
            of reward system to be used in the instantiated environment.
        :param reward_function: The `RewardFunction` function whose goal is to help compute the
            rewards in accordance with the selected (sub)type of reward system. It plays the role
            of either the ``graph_invariant`` or the ``graph_invariant_difference`` function from
            the description of the `RewardType` enumeration, and its expected signature varies
            depending on the ``reward_type`` argument.
        :param graph_order: A positive integer (not below two) that represents the graph order of
            the graphs to be constructed.
        :param flattened_ordering: An item of the `FlattenedOrdering` enumeration that determines
            whether the edges (resp. arcs) should be colored in the flattened row-major order or
            the flattened clockwise order. The default value is `FlattenedOrdering.ROW_MAJOR`,
            i.e., the edges (resp. arcs) should be colored in the flattened row-major order by
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
        """

        super().__init__(reward_type=reward_type, reward_function=reward_function)

        self._edge_colors: int = edge_colors
        self._is_directed: bool = is_directed
        self._allow_loops: bool = allow_loops
        self._flattened_ordering: FlattenedOrdering = flattened_ordering

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

        self._state_length: int = self._edge_colors * self._flattened_length
        self._step_count: Optional[int] = None

    def reset_batch(self, batch_size: int) -> Tuple[np.ndarray, EpisodeStatus]:
        self._state_batch = np.zeros((batch_size, self._state_length), dtype=np.uint8)
        # The zeroth edge (resp. arc) should be colored first.
        self._state_batch[:, -self._flattened_length] = 1
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
        """

        # If the graphs have only two proper edge colors, then the transition can easily be done as
        # follows.
        if self._edge_colors == 2:
            self._state_batch[:, self._step_count] = action_batch[:, 0]
        # Otherwise, the next trick should be used.
        else:
            view = self._state_batch.reshape(-1, self._edge_colors, self._flattened_length)
            rows = np.arange(self._state_batch.shape[0], dtype=np.int32)
            view[rows, action_batch[:, 0] - 1, self._step_count] = 1

        # Set the current edge (resp. arc) position flags to zero and increment the ``_step_count``
        # attribute.
        self._state_batch[:, -self._flattened_length + self._step_count] = 0
        self._step_count += 1

        if self._step_count < self._flattened_length:
            # Set the new current edge (resp. arc) position flags to one.
            self._state_batch[:, -self._flattened_length + self._step_count] = 1
        else:
            self._status = EpisodeStatus.TERMINATED

    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> GraphBatch:
        temp = state_batch.reshape(-1, self._edge_colors, self._flattened_length)

        # Extract the flattened format for all the graphs from the states, so that the edge color
        # $k$ for the uncolored edges (resp. arcs) is incorrectly represented by 0, while
        # everything else is correct.
        if self._edge_colors == 2:
            result = state_batch[:, : self._flattened_length]
        else:
            color_indices = np.arange(1, self._edge_colors, dtype=np.uint8)
            result = (temp[:, :-1, :] * color_indices[:, None]).sum(axis=1)

        # Fix the uncolored edges (resp. arcs) by coloring them with the color $k$, as required.
        uncolored_mask = np.maximum.accumulate(temp[:, -1, :], axis=1).astype(bool)
        result[uncolored_mask] = self._edge_colors

        return GraphBatch.from_flattened(
            flattened=result,
            flattened_ordering=self._flattened_ordering,
            edge_colors=self._edge_colors,
            is_directed=self._is_directed,
            allow_loops=self._allow_loops,
        )


class LinearSetEnvironment(GraphEnvironment):
    """
    This class inherits from the `GraphEnvironment` class and models a graph-building game in which
    the edges (resp. arcs) are all initially fully colored in some manner and are then recolored
    one by one, either in the flattened row-major order or the flattened clockwise order. The user
    can select the graph order and the number of proper edge colors, as well as choose whether the
    graphs should be directed or undirected, and whether loops should be allowed. Additionally, the
    user can configure the mechanism that controls how the initial fully colored graphs are
    colored, which could be both deterministic or nondeterministic. The edges (resp. arcs) are then
    recolored in the same way that a graph is reconstructed from a flattened format. In other
    words, if loops are not allowed, then these edges (resp. arcs) are just deleted and completely
    ignored, and if the graphs are undirected, then only the upper triangular part of an adjacency
    matrix is considered (with or without the diagonal, depending on whether loops are allowed) and
    each edge is essentially recolored in both directions at the same time. The user can choose
    whether the edges (resp. arcs) should be recolored in the flattened row-major order or the
    flattened clockwise order.

    The RL tasks in this environment are episodic, and the total number of actions to be performed
    equals the number of entries in each of the two flattened graph formats. Therefore, this number
    depends on the selected graph order, on whether the constructed graphs are directed, and on
    whether loops are allowed.

    Each state is represented by a binary `numpy.ndarray` list of type `numpy.uint8` and length
    ``edge_colors * flattened_length``, where ``edge_colors`` is the configured number of proper
    edge colors, while ``flattened_length`` is the length of the `numpy.ndarray` list from any of
    the two flattened graph formats of the graphs to be constructed. In the state vector, in the
    first ``flattened_length`` bits, the value 1 indicates which of the ``flattened_length`` edges
    (resp. arcs) is currently of the color 1; in the second ``flattened_length`` bits, the value 1
    indicates which of the ``flattened_length`` edges (resp. arcs) is currently of the color 2; and
    so on until the ``(edge_colors - 1)``-th ``flattened_length`` bits, where the value 1 indicates
    which of the ``flattened_length`` edges (resp. arcs) is currently of the color
    ``edge_colors - 1``. The final ``flattened_length`` bits represent a one-hot encoding of the
    position that determines the next edge (resp. arc) to be recolored. In other words, there is
    either one value of 1 whose index determines which edge (resp. arc) should be recolored next,
    or all the values are 0 and this signifies a terminal state, i.e., a state where all the edges
    (resp. arcs) have been recolored.

    Each action is represented by a single-entry `numpy.ndarray` list of type `numpy.int8`
    containing a value between 0 and ``edge_colors - 1`` that determines which color the next edge
    (resp. arc) should be recolored with.

    :ivar _edge_colors: The number of proper edge colors in the graphs to be constructed.
    :ivar _is_directed: A boolean that indicates whether the graphs to be constructed are a
        $k$-edge-colored looped complete directed graph or a $k$-edge-colored looped complete
        undirected graph.
    :ivar _allow_loops: A boolean that indicates whether the graphs to be constructed are allowed
        to have loops.
    :ivar _flattened_ordering: An item of the `FlattenedOrdering` enumeration that determines
        whether the edges (resp. arcs) should be recolored in the flattened row-major order or the
        flattened clockwise order.
    :ivar initial_graph_generator: A `GraphGenerator` function that describes the mechanism for
        generating the underlying fully colored graphs for the states in the batch of initial
        states.
    :ivar _flattened_length: A positive integer that determines the length of the `numpy.ndarray`
        list from any of the two flattened graph formats of the graphs to be constructed. This
        number also represents how many actions are needed in total to reach a terminal state from
        an initial state.
    :ivar _state_length: A positive integer that determines the length of each of the state
        vectors, i.e., the number ``_edge_colors * _flattened_length``.
    :ivar _step_count: An integer between 0 and ``_flattened_length`` that signifies the index of
        the next edge (resp. arc) to be recolored, in the order determined by the
        ``_flattened_ordering`` attribute. The value of ``_flattened_length`` indicates that all
        the edges (resp. arcs) have been recolored and that no edge (resp. arc) should be colored
        next, i.e., a terminal state has been reached.
    """

    def __init__(
        self,
        reward_type: RewardType,
        reward_function: RewardFunction,
        graph_order: int,
        flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
        edge_colors: int = 2,
        is_directed: bool = False,
        allow_loops: bool = False,
        initial_graph_generator: Optional[GraphGenerator] = None,
    ):
        """
        This constructor initializes an instance of the `LinearSetEnvironment` object.

        :param reward_type: An item of the `RewardType` enumeration that determines the (sub)type
            of reward system to be used in the instantiated environment.
        :param reward_function: The `RewardFunction` function whose goal is to help compute the
            rewards in accordance with the selected (sub)type of reward system. It plays the role
            of either the ``graph_invariant`` or the ``graph_invariant_difference`` function from
            the description of the `RewardType` enumeration, and its expected signature varies
            depending on the ``reward_type`` argument.
        :param graph_order: A positive integer (not below two) that represents the graph order of
            the graphs to be constructed.
        :param flattened_ordering: An item of the `FlattenedOrdering` enumeration that determines
            whether the edges (resp. arcs) should be recolored in the flattened row-major order or
            the flattened clockwise order. The default value is `FlattenedOrdering.ROW_MAJOR`,
            i.e., the edges (resp. arcs) should be recolored in the flattened row-major order by
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

        self._state_length: int = self._edge_colors * self._flattened_length
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
        # make sure that the zeroth edge (resp. arc) position flag is set to one.
        if self._edge_colors == 2:
            self._state_batch = np.zeros((batch_size, self._state_length), dtype=np.uint8)
            self._state_batch[:, : self._flattened_length] = format_representation
            self._state_batch[:, self._flattened_length] = 1
        else:
            color_indices = np.arange(1, self._edge_colors + 1, dtype=np.uint8)
            temp = (format_representation[:, None, :] == color_indices[:, None]).astype(np.uint8)
            temp[:, self._edge_colors - 1, :] = 0
            temp[:, self._edge_colors - 1, 0] = 1

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
            current states, given as a `numpy.ndarray` matrix of type `numpy.int8` where the rows
            correspond to the actions. The number of actions in this batch must be the same as the
            number of states in the `_state_batch` attribute.
        """

        # If the graphs have only two proper edge colors, then the transition can easily be done as
        # follows.
        if self._edge_colors == 2:
            self._state_batch[:, self._step_count] = action_batch[:, 0]
        # Otherwise, the next trick should be used.
        else:
            view = self._state_batch.reshape(-1, self._edge_colors, self._flattened_length)
            view[:, :, self._step_count] = 0
            rows = np.arange(self._state_batch.shape[0], dtype=np.int32)
            view[rows, action_batch[:, 0] - 1, self._step_count] = 1

        # Set the current edge (resp. arc) position flags to zero and increment the ``_step_count``
        # attribute.
        self._state_batch[:, -self._flattened_length + self._step_count] = 0
        self._step_count += 1

        if self._step_count < self._flattened_length:
            # Set the new current edge (resp. arc) position flags to one.
            self._state_batch[:, -self._flattened_length + self._step_count] = 1
        else:
            self._status = EpisodeStatus.TERMINATED

    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> GraphBatch:
        if self._edge_colors == 2:
            return GraphBatch.from_flattened(
                flattened=state_batch[:, : self._flattened_length],
                flattened_ordering=self._flattened_ordering,
                edge_colors=self._edge_colors,
                is_directed=self._is_directed,
                allow_loops=self._allow_loops,
            )

        temp = state_batch.reshape(-1, self._edge_colors, self._flattened_length)
        color_indices = np.arange(1, self._edge_colors, dtype=np.uint8)
        result = (temp[:, :-1, :] * color_indices[:, None]).sum(axis=1)

        return GraphBatch.from_flattened(
            flattened=result,
            flattened_ordering=self._flattened_ordering,
            edge_colors=self._edge_colors,
            is_directed=self._is_directed,
            allow_loops=self._allow_loops,
        )


class LinearFlipEnvironment(GraphEnvironment):
    """
    This class inherits from the `GraphEnvironment` class and models a graph-building game used for
    constructing 2-edge-colored looped complete graphs where the edges (resp. arcs) are all
    initially fully colored in some manner and are then potentially flipped, one by one, either in
    the flattened row-major order or the flattened clockwise order. The user can select the graph
    order, as well as choose whether the graphs should be directed or undirected, and whether loops
    should be allowed. Additionally, the user can configure the mechanism that controls how the
    initial fully colored graphs are colored, which could be both deterministic or
    nondeterministic. The edges (resp. arcs) are then potentially flipped in the same way that a
    graph is reconstructed from a flattened format. In other words, if loops are not allowed, then
    these edges (resp. arcs) are just deleted and completely ignored, and if the graphs are
    undirected, then only the upper triangular part of an adjacency matrix is considered (with or
    without the diagonal, depending on whether loops are allowed) and each edge is essentially
    potentially flipped in both directions at the same time. The user can choose whether the edges
    (resp. arcs) should be potentially flipped in the flattened row-major order or the flattened
    clockwise order.

    The RL tasks in this environment are episodic, and the total number of actions to be performed
    equals the number of entries in each of the two flattened graph formats. Therefore, this number
    depends on the selected graph order, on whether the constructed graphs are directed, and on
    whether loops are allowed.

    Each state is represented by a binary `numpy.ndarray` list of type `numpy.uint8` and length
    ``2 * flattened_length``, where ``flattened_length`` is the length of the `numpy.ndarray` list
    from any of the two flattened graph formats of the graphs to be constructed. In the state
    vector, in the first ``flattened_length`` bits, the value 1 indicates which of the
    ``flattened_length`` edges (resp. arcs) is currently of the color 1, while the final
    ``flattened_length`` bits represent a one-hot encoding of the position that determines the next
    edge (resp. arc) to be potentially flipped. In other words, there is either one value of 1
    whose index determines which edge (resp. arc) should be potentially flipped next, or all the
    values are 0 and this signifies a terminal state, i.e., a state where all the edges (resp.
    arcs) have been traversed.

    Each action is represented by a single-entry `numpy.ndarray` list of type `numpy.int8`
    containing a binary value that determines whether the next edge (resp. arc) should be flipped.
    In other words, the number 1 indicates that the proper edge color of the next edge (resp. arc)
    should be changed, while the number 0 indicates that the proper edge color of the next edge
    (resp. arc) should stay the same.

    :ivar _is_directed: A boolean that indicates whether the graphs to be constructed are a
        2-edge-colored looped complete directed graph or a 2-edge-colored looped complete
        undirected graph.
    :ivar _allow_loops: A boolean that indicates whether the graphs to be constructed are allowed
        to have loops.
    :ivar _flattened_ordering: An item of the `FlattenedOrdering` enumeration that determines
        whether the edges (resp. arcs) should be potentially flipped in the flattened row-major
        order or the flattened clockwise order.
    :ivar initial_graph_generator: A `GraphGenerator` function that describes the mechanism for
        generating the underlying fully colored graphs for the states in the batch of initial
        states.
    :ivar _flattened_length: A positive integer that determines the length of the `numpy.ndarray`
        list from any of the two flattened graph formats of the graphs to be constructed. This
        number also represents how many actions are needed in total to reach a terminal state from
        an initial state.
    :ivar _step_count: An integer between 0 and ``_flattened_length`` that signifies the index of
        the next edge (resp. arc) to be potentially flipped, in the order determined by the
        ``_flattened_ordering`` attribute. The value of ``_flattened_length`` indicates that
        all the edges (resp. arcs) have been traversed and that no edge (resp. arc) should be
        potentially flipped next, i.e., a terminal state has been reached.
    """

    def __init__(
        self,
        reward_type: RewardType,
        reward_function: RewardFunction,
        graph_order: int,
        flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
        is_directed: bool = False,
        allow_loops: bool = False,
        initial_graph_generator: Optional[GraphGenerator] = None,
    ):
        """
        This constructor initializes an instance of the `LinearFlipEnvironment` object.

        :param reward_type: An item of the `RewardType` enumeration that determines the (sub)type
            of reward system to be used in the instantiated environment.
        :param reward_function: The `RewardFunction` function whose goal is to help compute the
            rewards in accordance with the selected (sub)type of reward system. It plays the role
            of either the ``graph_invariant`` or the ``graph_invariant_difference`` function from
            the description of the `RewardType` enumeration, and its expected signature varies
            depending on the ``reward_type`` argument.
        :param graph_order: A positive integer (not below two) that represents the graph order of
            the graphs to be constructed.
        :param flattened_ordering: An item of the `FlattenedOrdering` enumeration that determines
            whether the edges (resp. arcs) should be potentially flipped in the flattened row-major
            order or the flattened clockwise order. The default value is
            `FlattenedOrdering.ROW_MAJOR`, i.e., the edges (resp. arcs) should be potentially
            flipped in the flattened row-major order by default.
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

        self._step_count: Optional[int] = None

    def reset_batch(self, batch_size: int) -> Tuple[np.ndarray, EpisodeStatus]:
        # Use the ``initial_graph_generator`` to generate the initial underlying fully colored
        # graphs.
        initial_graph_batch = self.initial_graph_generator(batch_size=batch_size)
        if self._flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            format_representation = initial_graph_batch.flattened_row_major
        else:
            format_representation = initial_graph_batch.flattened_clockwise

        self._state_batch = np.zeros((batch_size, 2 * self._flattened_length), dtype=np.uint8)
        self._state_batch[:, : self._flattened_length] = format_representation
        # The zeroth edge (resp. arc) should be colored first.
        self._state_batch[:, self._flattened_length] = 1

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
        """

        self._state_batch[:, self._step_count] ^= action_batch[:, 0]
        # Set the current edge (resp. arc) position flags to zero and increment the ``_step_count``
        # attribute.
        self._state_batch[:, self._flattened_length + self._step_count] = 0
        self._step_count += 1

        if self._step_count < self._flattened_length:
            # Set the new current edge (resp. arc) position flags to one.
            self._state_batch[:, self._flattened_length + self._step_count] = 1
        else:
            self._status = EpisodeStatus.TERMINATED

    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> GraphBatch:
        return GraphBatch.from_flattened(
            flattened=state_batch[:, : self._flattened_length],
            flattened_ordering=self._flattened_ordering,
            is_directed=self._is_directed,
            allow_loops=self._allow_loops,
        )
