"""
This ``Python`` module contains the `GraphEnvironment` class, which encapsulates the concept of a
reinforcement learning environment to be used in graph theory applications, alongside various other
associated classes and enumerations.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np

from ..graphs.graph import GraphBatch


# Although this class is just a wrapper around a `np.ndarray`, it is implemented for the sake of
# clarity and conciseness.
class StateBatch:
    """
    This class encapsulates the concept of a batch of states in the context of a reinforcement
    learning environment to be used in graph theory applications. It essentially behaves as a
    wrapper around a `np.ndarray` matrix whose rows correspond to the states from a given batch.

    :ivar __state_batch: The given batch of states, represented as a `np.ndarray` matrix. The
        matrix rows correspond to the states from the batch, while the number of columns depends on
        the concrete state space that the states are being taken from. This matrix has at least one
        row and at least one column.
    """

    def __init__(self, state_batch: np.ndarray):
        """
        This constructor initializes a `StateBatch` object that corresponds to the provided batch
        of states, given as a `np.ndarray` matrix.

        :param state_batch: The provided batch of states, given as a `np.ndarray` matrix. The
            matrix rows correspond to the states from the batch, while the number of columns
            depends on the concrete state space that the states are being taken from. The matrix
            must have at least one row and at least one column.
        """

        self.__state_batch: np.ndarray = state_batch

    @property
    def data(self) -> np.ndarray:
        """
        This property returns the `np.ndarray` matrix that represents the given batch of states.
        The matrix rows correspond to the states from the batch, while the number of columns
        depends on the concrete state space that the states are being taken from. The matrix has at
        least one row and at least one column.
        """

        return self.__state_batch


# Although this class is just a wrapper around a `np.ndarray`, it is implemented for the sake of
# clarity and conciseness.
class ActionBatch:
    """
    This class encapsulates the concept of a batch of actions in the context of a reinforcement
    learning environment to be used in graph theory applications. It essentially behaves as a
    wrapper around a `np.ndarray` matrix whose rows correspond to the actions from a given batch.

    :ivar __action_batch: The given batch of actions, represented as a `np.ndarray` matrix. The
        matrix rows correspond to the actions from the batch, while the number of columns depends
        on the concrete action space that the actions are being taken from. This matrix has at
        least one row and at least one column.
    """

    def __init__(self, action_batch: np.ndarray):
        """
        This constructor initializes an `ActionBatch` object that corresponds to the provided batch
        of actions, given as a `np.ndarray` matrix.

        :param action_batch: The provided batch of actions, given as a `np.ndarray` matrix. The
            matrix rows correspond to the actions from the batch, while the number of columns
            depends on the concrete action space that the actions are being taken from. The matrix
            must have at least one row and at least one column.
        """

        self.__action_batch: np.ndarray = action_batch

    @property
    def data(self) -> np.ndarray:
        """
        This property returns the `np.ndarray` matrix that represents the given batch of actions.
        The matrix rows correspond to the actions from the batch, while the number of columns
        depends on the concrete action space that the actions are being taken from. The matrix has
        at least one row and at least one column.
        """

        return self.__action_batch


# Although this class is just a wrapper around a `np.ndarray`, it is implemented for the sake of
# clarity and conciseness.
class RewardBatch:
    """
    This class encapsulates the concept of a batch of rewards in the context of a reinforcement
    learning environment to be used in graph theory applications. It essentially behaves as a
    wrapper around a `np.ndarray` list whose elements correspond to the rewards from a given batch.

    :ivar __reward_batch: The given batch of rewards, represented as a `np.ndarray` list whose
        elements correspond to the rewards from the batch. This list has at least one element.
    """

    def __init__(self, reward_batch: np.ndarray):
        """
        This constructor initializes a `RewardBatch` object that corresponds to the provided batch
        of rewards, given as a `np.ndarray` list.

        :param reward_batch: The provided batch of rewards, given as a `np.ndarray` list whose
            elements correspond to the rewards from the batch. The list must have at least one
            element, i.e., it cannot be empty.
        """

        self.__reward_batch: np.ndarray = reward_batch

    @property
    def data(self) -> np.ndarray:
        """
        This property returns the `np.ndarray` list that represents the given batch of rewards.
        The list is nonempty and its elements correspond to the rewards from the batch.
        """

        return self.__reward_batch


class RewardType(Enum):
    """
    This enumeration represents all the possible types of reward systems in the context of a
    reinforcement learning environment to be used in graph theory applications. In these
    applications, multiple episodes are acted on in parallel and all of their respective rewards
    are, therefore, also computed in parallel.

    :cvar SPARSE:
        A single actual reward is obtained at the end of an episode, while all the
        previously issued rewards are just equal to zero. The said reward is determined by the
        expression ``graph_invariant(terminal_graph_batch)``, where:

        * ``terminal_graph_batch`` is the underlying batch of graphs corresponding to the batch of
          terminal states; and
        * ``graph_invariant`` is a function that accepts a batch of graphs and returns the
          corresponding values for the graph invariant that is supposed to get maximized.

    :cvar TELESCOPIC:
        In each episode, a reward is issued after every action and it is computed by
        the formula ``graph_invariant(new_graph_batch) - graph_invariant(old_graph_batch)``, where:

        * ``new_graph_batch`` is the underyling batch of graphs corresponding to the batch of newly
          obtained states;
        * ``old_graph_batch`` is the underlying batch of graphs corresponding to the batch of
          previous states; and
        * ``graph_invariant`` is a function that accepts a batch of graphs and returns the
          corresponding values for the graph invariant that is supposed to get maximized.

    :cvar PROPER:
        In each episode, a reward is issued after every action and it is computed by
        the formula ``reward_function(old_graph_batch, new_graph_batch)``, where:

        * ``new_graph_batch`` is the underyling batch of graphs corresponding to the batch of newly
          obtained states;
        * ``old_graph_batch`` is the underlying batch of graphs corresponding to the batch of
          previous states; and
        * ``reward_function`` is a function that accepts a batch of previous graphs and a batch of
          new graphs, and returns the corresponding values for the element-wise differences of
          the graph invariant that is supposed to get maximized (the $i$-th element equals the
          graph invariant for the $i$-th new group minus the graph invariant for the $i$-th old
          graph).
    """

    SPARSE = 0
    TELESCOPIC = 1
    PROPER = 2


class EpisodeStatus(Enum):
    """
    This enumeration represents all the possible statuses that an episode could have in the context
    of a reinforcement learning environment to be used in graph theory applications.

    :cvar IN_PROGRESS: The episode is in progress, which means that it is in a state that accepts
        further actions.
    :cvar COMPLETED: The episode has reached a state whose underlying graph achieves satisfactory
        results, hence the reinforcement learning process should be stopped and the problem can be
        considered solved.
    :cvar TERMINATED: The episode has ended due to reaching an absorbing state, hence the
        environment cannot accept any further actions.
    :cvar TRUNCATED: The episode has ended since the required number of steps has been taken.
        Although the current state is not absorbing, no further actions should be performed.

    :note: This enumeration is also applicable to batches of episodes. In this case, a batch of
        episodes has the status `EpisodeStatus.COMPLETED` if at least one of its episodes has this
        status. Otherwise, it is guaranteed that all the episodes must enter an absorbing state at
        the same time, so bearing in mind that the steps are taken in all the episodes in parallel,
        this means that all the episodes have the same status. This common status is then
        considered the status of the given batch of episodes.
    """

    IN_PROGRESS = 0
    COMPLETED = 1
    TERMINATED = 2
    TRUNCATED = 3


class GraphEnvironment(ABC):
    """
    This abstract class encapsulates the concept of a reinforcement learning environment to be used
    in graph theory applications. For the sake of efficiency, it provides support for multiple
    episodes to be run in parallel. Concrete classes that inherit this abstract class must
    implement the following three abstract methods:

    * `reset_batch`, which serves to initialize a batch of episodes with a given batch size;
    * `_transition_batch`, which determines the transition process between states depending on the
      action taken; and
    * `state_batch_to_graph_batch`, which determines how the underlying graphs are extracted from
      states, i.e., how an underlying batch of graphs is extracted from a given batch of states.

    :ivar __reward_type: An item of the `RewardType` enumeration that determines the type of reward
        system that is used in the given environment.
    :ivar __reward_function: The function that helps compute the rewards in accordance with the
        selected type of reward system. It plays the role of either the ``graph_invariant`` or
        ``reward_function`` function from the description of the `RewardType` enumeration, and its
        expected signature varies depending on the selected type of reward system.
    :ivar _state_batch: Either `None`, or a `StateBatch` object that determines the batch of
        current states corresponding to the batch of episodes that are being run in parallel. This
        attribute is initially set to `None`, and afterwards, it is assigned the necessary
        `StateBatch` object after each invocation of the `reset_batch` or `step_batch` method.
    :ivar _status: Either `None`, or an item of the `EpisodeStatus` enumeration that signifies the
        status of the given batch of episodes, as described in the `EpisodeStatus` enumeration.
        This attribute is initially set to `None`, and afterwards, it is assigned the necessary
        `EpisodeStatus` enumeration item after each invocation of the `reset_batch` or `step_batch`
        method.
    """

    def __init__(self, reward_type: RewardType, reward_function: Callable):
        """
        This constructor initializes an instance of the `GraphEnvironment` object with a provided
        type of reward system and a corresponding function that helps compute the rewards.

        :param reward_type: An item of the `RewardType` enumeration that determines the type of
            reward system to be used in the instantiated environment.
        :param reward_function: The function whose goal is to help compute the rewards in
            accordance with the selected type of reward system. It plays the role of either the
            ``graph_invariant`` or ``reward_function`` function from the description of the
            `RewardType` enumeration, and its expected signature varies depending on the
            ``reward_type`` argument.
        """

        self.__reward_type: RewardType = reward_type
        self.__reward_function: Callable = reward_function

        self._state_batch: Optional[np.ndarray] = None
        self._status: Optional[EpisodeStatus] = None

    @abstractmethod
    def reset_batch(self, batch_size: int) -> Tuple[np.ndarray, EpisodeStatus]:
        """
        This abstract method must be implemented in any concrete class that inherits the
        `GraphEnvironment` class. It should initialize a batch of episodes with a given batch size,
        and update the `_state_batch` and `_status` attributes accordingly. The function should
        return the obtained batch of corresponding states after the initialization has been done,
        i.e., the value of the `_state_batch` attribute, as well as the status corresponding to the
        initialized batch of episodes, i.e., the value of the `_status` attribute.

        :param batch_size: The batch size of the batch of episodes that should be initialized,
            i.e., the number of episodes in it, given as a positive integer.

        :return: A tuple ``(initial_state_batch, status)``, where

            * ``initial_state_batch`` is the value of the `_state_batch` attribute after the batch
              of episodes has been initialized, given as a `StateBatch` object; and
            * ``status`` is the value of the `_status` attribute after the batch of episodes has
              been initialized, given as an item of the `EpisodeStatus` enumeration.
        """

        pass

    def step_batch(self, action_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, EpisodeStatus]:
        """
        This method takes a batch of actions and applies them element-wise to the states from the
        batch of current states given by the `_state_batch` attribute. More precisely, these two
        batches must be of the same size, and the $i$-th action should be applied to the $i$-th
        state. The method returns a batch of new states obtained after the actions have been
        performed, alongside the computed rewards and the new status corresponding to the batch of
        episodes run in parallel. Here, the order of the new states and the rewards in their
        respective batches matches the order of the performed actions and the original states.

        :param action_batch: The batch of actions to be applied to the states from the current
            batch of states, given as an `ActionBatch` object. The number of actions from this
            batch must be the same as the number of states from the `_state_batch` attribute.

        :return: A tuple ``(new_state_batch, reward_batch, status)``, where:

            * ``new_state_batch`` is the batch of newly obtained states, given as a `StateBatch`
              object;
            * ``reward_batch`` is the batch of computed rewards, given as a `RewardBatch` object;
              and
            * ``status`` is an item of the `EpisodeStatus` enumeration that determines the new
              status corresponding to the batch of episodes run in parallel.
        """

        new_state_batch, status = self._transition_batch(action_batch)
        reward_batch = None

        if self.__reward_type == RewardType.SPARSE:
            if status == EpisodeStatus.IN_PROGRESS:
                reward_batch = np.zeros((new_state_batch.shape[0],), dtype=float)
            else:
                new_underlying_graph_batch = self.state_batch_to_graph_batch(new_state_batch)
                reward_batch = self.__reward_function(new_underlying_graph_batch)

        else:
            new_underlying_graph_batch = self.state_batch_to_graph_batch(new_state_batch)
            old_underyling_graph_batch = self.state_batch_to_graph_batch(self._state_batch)

            if self.__reward_type == RewardType.TELESCOPIC:
                reward_batch = self.__reward_function(
                    new_underlying_graph_batch
                ) - self.__reward_function(old_underyling_graph_batch)
            else:
                reward_batch = self.__reward_function(
                    old_underyling_graph_batch, new_underlying_graph_batch
                )

        self._state_batch = new_state_batch
        self._status = status

        return self._state_batch, reward_batch, status

    @abstractmethod
    def _transition_batch(self, action_batch: np.ndarray) -> Tuple[np.ndarray, EpisodeStatus]:
        """
        This abstract method must be implemented in any concrete class that inherits the
        `GraphEnvironment` class. It should determine which new batch of states should be entered
        after the provided batch of actions is applied element-wise to the states from the batch of
        current states given by the `_state_batch` attribute. The function should return this batch
        of obtained states, alongside the new status corresponding to the batch of episodes run in
        parallel. Here, the order of the new states in the returned batch should match the order of
        the performed actions and the original states.

        :param action_batch: The batch of actions to be applied to the states from the current
            batch of states, given as an `ActionBatch` object. The number of actions from this
            batch must be the same as the number of states from the `_state_batch` attribute.

        :return: A tuple ``(new_state_batch, status)``, where:

            * ``new_state_batch`` is the batch of newly obtained states, given as a `StateBatch`
              object; and
            * ``status`` is an item of the `EpisodeStatus` enumeration that determines the new
              status corresponding to the batch of episodes run in parallel.

        :note: The implementation of this method must not modify the attributes `_state_batch` and
            `_status`. The obtained batch of new states and the corresponding status should just be
            returned without updating these two attributes since the logic behind this is located
            in the non-abstract `step_batch` method.
        """

        pass

    @abstractmethod
    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> GraphBatch:
        """
        This abstract method must be implemented in any concrete class that inherits the
        `GraphEnvironment` class. Its goal is to extract the batch of underlying graphs from any
        provided batch of states. The graphs should appear in the same order as their corresponding
        states in the given batch of states.

        :param state_batch: The batch of provided states whose underlying graphs should be
            extracted, given as a `StateBatch` object.

        :return: The extracted batch of underlying graphs, given as a `GraphBatch` object.

        :note: The implementation of this method must be a pure function, i.e., it should not
            modify any attributes of the given instance.
        """

        pass
