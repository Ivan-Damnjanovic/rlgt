"""
This ``Python`` module defines the `GraphEnvironment` abstract class, which encapsulates the
concept of a reinforcement learning (RL) environment for graph theory applications, together with
an associated enumeration describing the possible episode statuses.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np

from ..graphs.graph import Graph


GraphInvariant = Callable[[Graph], np.ndarray]
"""
This is a type alias for functions that compute the values of a graph invariant for a given batch
of graphs. Such functions accept a `Graph` object representing a batch of graphs and return a
one-dimensional `numpy.ndarray` of type `numpy.float32`, whose length equals the batch size and
whose entries are the computed graph invariant values.
"""


GraphInvariantDiff = Callable[[Graph, Graph], np.ndarray]
"""
This is a type alias for functions that compute element-wise differences of a given graph invariant
between two batches of graphs. Such functions accept two `Graph` objects representing the two
batches and return a one-dimensional `numpy.ndarray` of type `numpy.float32`.

Specifically, the $i$-th entry of the output must equal the value of the graph invariant for the
$i$-th graph in the second batch minus the value of the same invariant for the $i$-th graph in the
first batch. The two batches must have the same size, and the corresponding graphs must have the
same order, the same number of proper edge colors, and the same graph type, meaning that they are
either all directed or all undirected, and either all allow loops or all disallow them.
"""


class EpisodeStatus(Enum):
    """
    This enumeration represents the possible statuses of an episode in a reinforcement learning
    environment for graph theory applications.

    :cvar IN_PROGRESS: The episode is ongoing, meaning that the environment may continue to accept
        actions.
    :cvar TERMINATED: The episode has ended because a terminal state has been reached, so the
        environment can no longer accept actions. This status arises only in environments with
        episodic tasks.
    :cvar TRUNCATED: The episode has ended because a predefined step limit has been reached. The
        current state is not terminal, but no further actions should be taken. This status arises
        only in environments with continuing tasks.

    :note: This enumeration also applies to batches of episodes. In episodic tasks, all parallel
        episodes are guaranteed to reach a terminal state after the same number of actions. As a
        result, regardless of whether the tasks are episodic or continuing, all episodes within a
        batch always share the same status. This common value is regarded as the status of the
        entire batch.
    """

    IN_PROGRESS = 0
    TERMINATED = 1
    TRUNCATED = 2


class GraphEnvironment(ABC):
    """
    This abstract class encapsulates the concept of a reinforcement learning (RL) environment for
    graph theory applications. Such an environment is designed to address extremal problems in
    which a specified graph invariant is to be maximized over a finite family of fully colored
    $k$-edge-colored looped complete graphs.

    States are represented as fixed-length `numpy.ndarray` vectors. Actions are represented as
    `numpy.int32` integers taking values in the range from 0 to ``action_number - 1``, where
    ``action_number`` denotes the total (finite) number of available actions. At each step, some
    actions may be unavailable, with the constraint that at least one action is always available in
    any non-terminal state.

    For efficiency, the environment supports running multiple episodes in parallel. All episodes
    are guaranteed to terminate after a predetermined number of steps, regardless of whether the
    underlying task is episodic or continuing. Accordingly, batches of states are represented as
    two-dimensional `numpy.ndarray` matrices whose rows correspond to individual states, while
    batches of actions are represented as one-dimensional `numpy.ndarray` vectors of type
    `numpy.int32`.

    Instead of returning rewards in the conventional RL sense, the environment returns values of a
    selected graph invariant associated with the underlying graphs corresponding to the newly
    reached states. The graph invariant is specified via a `GraphInvariant` function. If the sparse
    setting is enabled, graph invariant values are computed only for the final batch of actions,
    and `None` is returned at all preceding steps. Otherwise, the invariant values are computed
    after every batch of actions. In the latter case, the computation may be optimized by supplying
    a `GraphInvariantDiff` function, which specifies how graph invariant values change when the
    environment transitions from one batch of underlying graphs to another.

    Concrete subclasses must implement the following abstract properties:

    1. `state_length`, which returns the length of the state vectors;
    2. `state_dtype`, which returns the data type of the state vectors;
    3. `action_number`, which returns the total number of available actions;
    4. `action_mask`, which specifies which actions are currently available;
    5. `episode_length`, which returns the predetermined length of each episode; and
    6. `is_continuing`, which determines whether the RL tasks in the environment are continuing or
       episodic.

    Concrete subclasses must also implement the following abstract methods:

    1. `_initialize_batch`, which initializes a batch of episodes;
    2. `_transition_batch`, which applies a batch of actions to the current batch of states; and
    3. `state_batch_to_graph_batch`, which extracts the underlying batch of graphs from a provided
       batch of states.

    :ivar __graph_invariant: A `GraphInvariant` function specifying the graph invariant to be
        maximized.
    :ivar __graph_invariant_diff: Either `None`, indicating that graph invariant values are always
        computed directly using `__graph_invariant`, or a `GraphInvariantDiff` function used to
        incrementally update invariant values after state transitions.
    :ivar sparse_setting: A `bool` indicating whether the graph invariant values should be computed
        only for the final batch of actions.
    :ivar _state_batch: Either `None` or a two-dimensional `numpy.ndarray` representing the current
        batch of states.
    :ivar _status: Either `None` or an `EpisodeStatus` value describing the current status of the
        batch of episodes.
    :ivar __graph_batch: Either `None` or a `Graph` object representing the current batch of
        underlying graphs. This attribute is updated only when required by the sparse setting.
    :ivar __graph_invariant_batch: Either `None` or a one-dimensional `numpy.ndarray` of type
        `numpy.float32` containing the current batch of graph invariant values. As with
        `__graph_batch`, this attribute is updated only when required by the sparse setting.
    """

    def __init__(
        self,
        graph_invariant: GraphInvariant,
        graph_invariant_diff: Optional[GraphInvariantDiff] = None,
        sparse_setting: bool = False,
    ):
        """
        This constructor initializes a `GraphEnvironment` with a specified graph invariant and,
        optionally, a function for computing differences of that invariant between successive
        batches of graphs.

        :param graph_invariant: A `GraphInvariant` function that computes the graph invariant
            values associated with a batch of underlying graphs. These values are the quantities to
            be maximized by the environment.
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

        self.__graph_invariant: Callable = graph_invariant
        self.__graph_invariant_diff: Optional[Callable] = graph_invariant_diff
        self.sparse_setting: bool = sparse_setting

        self._state_batch: Optional[np.ndarray] = None
        self._status: Optional[EpisodeStatus] = None

        self.__graph_batch: Optional[Graph] = None
        self.__graph_invariant_batch: Optional[np.ndarray] = None

    @property
    @abstractmethod
    def state_length(self) -> int:
        """
        This abstract property must be implemented by any concrete subclass. It must return the
        number of entries in each state vector, i.e., the length of the one-dimensional
        `numpy.ndarray` vectors that represent states, as a positive `int`.
        """

        pass  # pragma: no cover

    @property
    @abstractmethod
    def state_dtype(self) -> np.dtype:
        """
        This abstract property must be implemented by any concrete subclass. It must return the
        data type of the one-dimensional `numpy.ndarray` vectors that represent states, as a
        `numpy.dtype`.
        """

        pass  # pragma: no cover

    @property
    @abstractmethod
    def action_number(self) -> int:
        """
        This abstract property must be implemented by any concrete subclass. It must return the
        total number of distinct actions that can be executed in the environment, as a positive
        `int`.
        """

        pass  # pragma: no cover

    @property
    @abstractmethod
    def action_mask(self) -> Optional[np.ndarray]:
        """
        This abstract property must be implemented by any concrete subclass. It must return `None`
        if no episodes are currently being run in parallel, or if every action is available in
        every current state. Otherwise, it must return a two-dimensional `numpy.ndarray` matrix
        ``a`` of type `bool` whose entry ``a[i, j]`` is `True` if and only if action $j$ is
        available in the current state of the $i$-th episode.
        """

        pass  # pragma: no cover

    @property
    @abstractmethod
    def episode_length(self) -> int:
        """
        This abstract property must be implemented by any concrete subclass. It must return the
        predetermined common length of all episodes run in parallel, i.e., the total number of
        actions executed in each episode, as a positive `int`.
        """

        pass  # pragma: no cover

    @property
    @abstractmethod
    def is_continuing(self) -> bool:
        """
        This abstract property must be implemented by any concrete subclass. It must return a
        `bool` indicating whether the RL tasks in the environment are continuing (`True`) or
        episodic (`False`).
        """

        pass

    def reset_batch(
        self, batch_size: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray], EpisodeStatus]:
        """
        This method initializes a batch of episodes of a specified size and returns the resulting
        batch of states, the corresponding values of the selected graph invariant (if computed),
        and the status of the batch of episodes. The order of the returned graph invariant values
        matches the order of the states in the initialized batch. If the sparse setting is enabled,
        the graph invariant values are not computed at initialization and `None` is returned in
        their place. Otherwise, the graph invariant values are computed immediately after
        initialization.

        :param batch_size: The number of episodes to initialize in the batch, given as a positive
            `int`.

        :return: A tuple ``(initial_state_batch, graph_invariant_batch, status)``, where

            * ``initial_state_batch`` is a two-dimensional `numpy.ndarray` whose rows represent the
              initial states of the individual episodes in the batch;
            * ``graph_invariant_batch`` is either a one-dimensional `numpy.ndarray` of type
              `numpy.float32` containing the graph invariant values corresponding to the
              initialized states, or `None` if the sparse setting is used; and
            * ``status`` is an `EpisodeStatus` item describing the status of the batch of episodes
              immediately after initialization.
        """

        self._initialize_batch(batch_size=batch_size)

        # If the sparse setting is not used, then update the batch of graphs and the corresponding
        # graph invariant values.
        if not self.sparse_setting:
            self.__graph_batch = self.state_batch_to_graph_batch(self._state_batch)
            self.__graph_invariant_batch = self.__graph_invariant(self.__graph_batch)

        return self._state_batch, self.__graph_invariant_batch, self._status

    @abstractmethod
    def _initialize_batch(self, batch_size: int) -> None:
        """
        This abstract method must be implemented by any concrete subclass. It must initialize a
        batch of episodes of the specified size and update the `_state_batch` and `_status`
        attributes so that they represent the newly initialized batch.

        :param batch_size: The number of episodes to initialize in the batch, given as a positive
            `int`.
        """

        pass

    def step_batch(
        self, action_batch: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], EpisodeStatus]:
        """
        This method applies a batch of actions to the current batch of episodes and returns the
        resulting batch of states, the corresponding values of the selected graph invariant (if
        computed), and the updated status of the batch. The $i$-th provided action is applied to
        the $i$-th state in ``_state_batch``. The order of the returned states and graph invariant
        values matches the order of the applied actions and the original states. If the sparse
        setting is enabled, the graph invariant values are computed only when a final state is
        reached. Otherwise, the graph invariant values are computed after every batch of actions,
        either directly or via the graph invariant differences function if one is provided.

        :param action_batch: A one-dimensional `numpy.ndarray` of type `numpy.int32` containing the
            actions to be applied. The length of ``action_batch`` must match the number of states in
            ``_state_batch``.

        :return: A tuple ``(new_state_batch, graph_invariant_batch, status)``, where

            * ``new_state_batch`` is a two-dimensional `numpy.ndarray` whose rows represent the
              states after the actions have been applied;
            * ``graph_invariant_batch`` is either a one-dimensional `numpy.ndarray` of type
              `numpy.float32` containing the corresponding graph invariant values, or `None` if
              such values are not computed at this step due to the sparse setting; and
            * ``status`` is an `EpisodeStatus` item describing the updated status of the batch of
              episodes.
        """

        # Disallow transitions once the batch of episodes has ended.
        if self._status != EpisodeStatus.IN_PROGRESS:
            raise RuntimeError(
                "Cannot execute actions since the batch of episodes is not in progress."
            )

        # If the sparse setting is used, then compute the graph invariant values only at the end of
        # the episode.
        if self.sparse_setting:
            self._transition_batch(action_batch)

            if self._status != EpisodeStatus.IN_PROGRESS:
                self.__graph_batch = self.state_batch_to_graph_batch(self._state_batch)
                self.__graph_invariant_batch = self.__graph_invariant(self.__graph_batch)

            return self._state_batch, self.__graph_invariant_batch, self._status

        # Settle the case where the sparse setting is not used, and there is no
        # invariant-difference optimization.
        if self.__graph_invariant_diff is None:
            self._transition_batch(action_batch)
            self.__graph_batch = self.state_batch_to_graph_batch(self._state_batch)
            self.__graph_invariant_batch = self.__graph_invariant(self.__graph_batch)

            return self._state_batch, self.__graph_invariant_batch, self._status

        # Settle the case where the sparse setting is not used, and there is invariant-difference
        # optimization.
        old_graph_batch = self.__graph_batch
        self._transition_batch(action_batch)
        self.__graph_batch = self.state_batch_to_graph_batch(self._state_batch)
        self.__graph_invariant_batch += self.__graph_invariant_diff(
            old_graph_batch, self.__graph_batch
        )

        return self._state_batch, self.__graph_invariant_batch, self._status

    @abstractmethod
    def _transition_batch(self, action_batch: np.ndarray) -> None:
        """
        This abstract method must be implemented by any concrete subclass. It must apply a batch of
        actions to the current batch of states and update the `_state_batch` and `_status`
        attributes to reflect the resulting states and the updated batch status. Implementations
        may also update additional subclass-specific attributes as required.

        :param action_batch: A one-dimensional `numpy.ndarray` of type `numpy.int32` containing the
            actions to be applied. The length of ``action_batch`` must match the number of states
            in ``_state_batch``.
        """

        pass  # pragma: no cover

    @abstractmethod
    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> Graph:
        """
        This abstract method must be implemented by any concrete subclass. It extracts the batch of
        underlying graphs corresponding to a provided batch of states. Implementations must return
        a `Graph` object containing the graphs corresponding to each row in `state_batch`,
        preserving the row order. This method must be pure and **must not modify** any attributes
        of the class instance.

        :param state_batch: A two-dimensional `numpy.ndarray` whose rows represent individual
            states from which the underlying graphs are to be extracted.

        :return: A `Graph` object representing the extracted batch of graphs.
        """

        pass  # pragma: no cover

    def state_to_graph(self, state: np.ndarray) -> Graph:  # pragma: no cover
        """
        This method extracts the underlying graph corresponding to a single state.

        :param state: A one-dimensional `numpy.ndarray` representing a single state.

        :return: The underlying graph corresponding to `state`, returned as a `Graph` object.

        :note: This method is pure and **does not modify** any attributes of the class instance. It
            internally calls `state_batch_to_graph_batch` with a singleton batch.
        """

        graph_batch = self.state_batch_to_graph_batch(state.reshape(1, -1))

        return graph_batch[0]
