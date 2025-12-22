"""
This ``Python`` module contains the `GraphEnvironment` abstract class, which encapsulates the
concept of a reinforcement learning environment to be used in graph theory applications, alongside
two associated enumerations concerning the types of reward systems and the episode statuses.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Optional, Tuple, Union

import numpy as np

from ..graphs.graph_batch import GraphBatch


RewardFunction = Union[
    Callable[[GraphBatch], np.ndarray],
    Callable[[GraphBatch, GraphBatch], np.ndarray],
]
"""
This is the type alias for the functions that help compute the rewards in RL environments to be
used in graph theory applications. In other words, these are the functions that take on the role of
either the ``graph_invariant`` or the ``reward_function`` function from the description of the
`RewardType` enumeration.
"""


class RewardType(Enum):
    """
    This enumeration represents the types and subtypes of reward systems in the context of an RL
    environment to be used in graph theory applications. In these applications, multiple episodes
    are acted on in parallel, and all of their respective rewards are, therefore, also computed in
    parallel. We recognize two types of reward systems: the sparse reward system and the dense
    reward system, which together contain three subtypes in total. The sparse reward system forms
    one of these subtypes on its own, while the dense reward system is divided into two separate
    subtypes: the telescopic reward system and the proper reward system, for implementational
    reasons.

    :cvar SPARSE:
        The sparse reward system. The batch of corresponding rewards received after every batch of
        actions is zero, apart from the final batch of actions, for which it is equal to
        ``graph_invariant(final_graph_batch)``, where:

        * ``final_graph_batch`` is the underlying batch of graphs corresponding to the batch of
          final states; and
        * ``graph_invariant`` is a function that accepts a batch of graphs and returns the
          corresponding values for the graph invariant that is supposed to get maximized.

    :cvar TELESCOPIC:
        The telescopic reward system. The batch of corresponding rewards received after each batch
        of actions is computed by the formula
        ``graph_invariant(new_graph_batch) - graph_invariant(old_graph_batch)``, where:

        * ``new_graph_batch`` is the underlying batch of graphs corresponding to the batch of newly
          obtained states;
        * ``old_graph_batch`` is the underlying batch of graphs corresponding to the batch of
          previous states; and
        * ``graph_invariant`` is a function that accepts a batch of graphs and returns the
          corresponding values for the graph invariant that is supposed to get maximized.

    :cvar PROPER:
        The proper reward system. The batch of corresponding rewards received after each batch of
        actions is computed by the formula ``reward_function(old_graph_batch, new_graph_batch)``,
        where:

        * ``new_graph_batch`` is the underlying batch of graphs corresponding to the batch of newly
          obtained states;
        * ``old_graph_batch`` is the underlying batch of graphs corresponding to the batch of
          previous states; and
        * ``reward_function`` is a function that accepts a batch of previous underlying graphs and
          a batch of new underlying graphs, and returns the corresponding values for the
          element-wise differences of the graph invariant that is supposed to get maximized (the
          $i$-th element equals the graph invariant for the $i$-th new underlying graph minus the
          graph invariant for the $i$-th old underlying graph).

    :note: The dense reward system is divided into the telescopic reward system and the proper
        reward system because, in certain cases, it can be more computationally efficient to invoke
        the ``reward_function`` function once than invoke the ``graph_invariant`` function twice.
    """

    SPARSE = 0
    TELESCOPIC = 1
    PROPER = 2


class EpisodeStatus(Enum):
    """
    This enumeration represents all the possible statuses that an episode can have in the context
    of an RL environment to be used in graph theory applications.

    :cvar IN_PROGRESS: The episode is in progress, which means that it is in a state that accepts
        further actions.
    :cvar TERMINATED: The episode has ended due to reaching a terminal state, hence the environment
        cannot accept any further actions. This status appears only in RL environments where the
        tasks are episodic.
    :cvar TRUNCATED: The episode has ended since the required number of steps has been taken. In
        this case, although the current state is not terminal, no further actions should be
        performed. This status appears only in RL environments where the tasks are continuing.

    :note: This enumeration is also applicable to batches of episodes. If the tasks are episodic,
        then the parallelized episodes are guaranteed to enter a terminal state after the same
        number of performed actions. Therefore, regardless of whether the tasks are episodic or
        continuing, all the episodes must always have the same status. This common status is then
        considered to be the status of the given batch of episodes.
    """

    IN_PROGRESS = 0
    TERMINATED = 1
    TRUNCATED = 2


class GraphEnvironment(ABC):
    """
    This abstract class encapsulates the concept of an RL environment to be used in graph theory
    applications. For the sake of efficiency, it provides support for multiple episodes to be run
    in parallel. This approach makes sense because it is guaranteed that all of these episodes must
    end at the same time, regardless of whether the RL environment has episodic or continuing
    tasks. Concrete classes that inherit from this abstract class must implement the following
    three abstract methods:

    * `reset_batch`, which serves to initialize a batch of episodes with a given batch size;
    * `_transition_batch`, which determines the transition process between states depending on the
      action taken; and
    * `state_batch_to_graph_batch`, which determines how the underlying graphs are extracted from
      states, i.e., how a batch of underlying graphs is extracted from a given batch of states.

    :ivar __reward_type: An item of the `RewardType` enumeration that determines the (sub)type of
        reward system that is used in the given RL environment.
    :ivar __reward_function: The `RewardFunction` function that helps compute the rewards in
        accordance with the selected (sub)type of reward system. It plays the role of either the
        ``graph_invariant`` or the ``reward_function`` function from the description of the
        `RewardType` enumeration, and its expected signature varies depending on the selected
        (sub)type of reward system.
    :ivar _state_batch: Either `None`, or a `np.ndarray` matrix that determines the batch of
        current states corresponding to the batch of episodes that are being run in parallel. This
        attribute is initially set to `None`, and afterwards, it is assigned the necessary
        `np.ndarray` matrix after each invocation of the `reset_batch` or `step_batch` method. The
        rows of the said `np.ndarray` matrix correspond to the states in the batch.
    :ivar _status: Either `None`, or an item of the `EpisodeStatus` enumeration that signifies the
        status of the given batch of episodes, as described in the `EpisodeStatus` enumeration.
        This attribute is initially set to `None`, and afterwards, it is assigned the necessary
        `EpisodeStatus` enumeration item after each invocation of the `reset_batch` or `step_batch`
        method.
    """

    def __init__(self, reward_type: RewardType, reward_function: RewardFunction):
        """
        This constructor initializes an instance of the `GraphEnvironment` object with a provided
        (sub)type of reward system and a corresponding function that helps compute the rewards.

        :param reward_type: An item of the `RewardType` enumeration that determines the (sub)type
            of reward system to be used in the instantiated environment.
        :param reward_function: The `RewardFunction` function whose goal is to help compute the
            rewards in accordance with the selected (sub)type of reward system. It plays the role
            of either the ``graph_invariant`` or the ``reward_function`` function from the
            description of the `RewardType` enumeration, and its expected signature varies
            depending on the ``reward_type`` argument.
        """

        self.__reward_type: RewardType = reward_type
        self.__reward_function: RewardFunction = reward_function

        self._state_batch: Optional[np.ndarray] = None
        self._status: Optional[EpisodeStatus] = None

    @abstractmethod
    def reset_batch(self, batch_size: int) -> Tuple[np.ndarray, EpisodeStatus]:
        """
        This abstract method must be implemented in any concrete class that inherits from the
        `GraphEnvironment` class. It should initialize a batch of episodes with a given batch size,
        and update the `_state_batch` and `_status` attributes accordingly. The function should
        return the obtained batch of corresponding states after the initialization has been
        completed, i.e., the value of the `_state_batch` attribute, as well as the status
        corresponding to the initialized batch of episodes, i.e., the value of the `_status`
        attribute.

        :param batch_size: The batch size of the batch of episodes that should be initialized,
            i.e., the number of episodes in it, given as a positive integer.

        :return: A tuple ``(initial_state_batch, status)``, where

            * ``initial_state_batch`` is the value of the `_state_batch` attribute after the batch
              of episodes has been initialized, given as a `np.ndarray` matrix where the rows
              corresponds to the states in the batch; and
            * ``status`` is the value of the `_status` attribute after the batch of episodes has
              been initialized, given as an item of the `EpisodeStatus` enumeration.
        """

        pass  # pragma: no cover

    def step_batch(self, action_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, EpisodeStatus]:
        """
        This method takes a batch of actions and applies them element-wise to the states in the
        batch of current states given by the `_state_batch` attribute. More precisely, these two
        batches must be of the same size, and the $i$-th action should be applied to the $i$-th
        state. The method returns a batch of new states obtained after the actions have been
        performed, alongside the computed rewards and the new status corresponding to the batch of
        episodes run in parallel. Here, the order of the new states and the rewards in their
        respective batches matches the order of the performed actions and the original states.

        :param action_batch: The batch of actions to be applied to the states in the batch of
            current states, given as a `np.ndarray` matrix where the rows correspond to the
            actions. The number of actions in this batch must be the same as the number of states
            in the `_state_batch` attribute.

        :return: A tuple ``(new_state_batch, reward_batch, status)``, where:

            * ``new_state_batch`` is the batch of newly obtained states, given as a `np.ndarray`
              matrix where the rows correspond to the states;
            * ``reward_batch`` is the batch of computed rewards, given as a `np.ndarray` list; and
            * ``status`` is an item of the `EpisodeStatus` enumeration that determines the new
              status corresponding to the batch of episodes run in parallel.
        """

        # Raise a `RuntimeError` if the user is trying to execute a batch of actions when the
        # episodes have already ended.
        if self._status != EpisodeStatus.IN_PROGRESS:
            raise RuntimeError

        # In this case, the sparse reward system is being used.
        if self.__reward_type == RewardType.SPARSE:
            # Execute the batch of actions and transition to the batch of new states.
            self._transition_batch(action_batch)

            # If the terminal states have not been reached, then just set all the rewards to zero.
            if self._status == EpisodeStatus.IN_PROGRESS:
                reward_batch = np.zeros((self._state_batch.shape[0],), dtype=float)
            # Otherwise, compute the rewards as the corresponding values for the graph invariant
            # that is supposed to get maximized. In other words, invoke the ``graph_invariant``
            # function.
            else:
                final_graph_batch = self.state_batch_to_graph_batch(self._state_batch)
                reward_batch = self.__reward_function(final_graph_batch)

        # In this case, the dense reward system is being used.
        else:
            old_graph_batch = self.state_batch_to_graph_batch(self._state_batch)
            # Execute the batch of actions and transition to the batch of new states.
            self._transition_batch(action_batch)
            new_graph_batch = self.state_batch_to_graph_batch(self._state_batch)

            # If the telescopic reward system is being used, then the rewards should be computed by
            # invoking the ``graph_invariant`` function twice.
            if self.__reward_type == RewardType.TELESCOPIC:
                reward_batch = self.__reward_function(new_graph_batch) - self.__reward_function(
                    old_graph_batch
                )
            # Otherwise, the proper reward system is being used, so the rewards should be computed
            # by invoking the ``reward_function`` function once.
            else:
                reward_batch = self.__reward_function(old_graph_batch, new_graph_batch)

        return self._state_batch, reward_batch, self._status

    @abstractmethod
    def _transition_batch(self, action_batch: np.ndarray) -> None:
        """
        This abstract method must be implemented in any concrete class that inherits from the
        `GraphEnvironment` class. It should perform the transition process by executing a batch of
        given actions and applying them element-wise to the states in the batch of current states
        given by the `_state_batch` attribute. The method should update the `_state_batch` and
        `_status` attributes, as well as any other potential attributes specific to the concrete
        class that inherits from the `GraphEnvironment` class.

        :param action_batch: The batch of actions to be applied to the states in the batch of
            current states, given as a `np.ndarray` matrix where the rows correspond to the
            actions. The number of actions in this batch must be the same as the number of states
            in the `_state_batch` attribute.
        """

        pass  # pragma: no cover

    @abstractmethod
    def state_batch_to_graph_batch(self, state_batch: np.ndarray) -> GraphBatch:
        """
        This abstract method must be implemented in any concrete class that inherits from the
        `GraphEnvironment` class. Its goal is to extract the batch of underlying graphs from any
        provided batch of states. The graphs should appear in the same order as their corresponding
        states in the given batch of states.

        :param state_batch: The batch of provided states whose underlying graphs should be
            extracted, given as a `np.ndarray` matrix whose rows correspond to the states in the
            batch.

        :return: The extracted batch of underlying graphs, given as a `GraphBatch` object.

        :note: The implementation of this method must be a pure function, i.e., it should not
            modify any attributes of the given instance.
        """

        pass  # pragma: no cover
