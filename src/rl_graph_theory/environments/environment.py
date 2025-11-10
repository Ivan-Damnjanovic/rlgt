"""
#TODO
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
    #TODO
    """

    CUMULATIVE = 0
    TELESCOPIC = 1
    INCREMENTAL = 2


class GraphEnvironment(ABC):
    """
    #TODO
    """

    def __init__(self, reward_type: RewardType, reward_function: Callable):
        self._reward_type: RewardType = reward_type
        self._reward_function: Callable = reward_function
        self._state_batch: Optional[StateBatch] = None

    @abstractmethod
    def reset_batch(self, batch_size: int) -> StateBatch:
        """
        #TODO
        """

        pass

    @abstractmethod
    def step_batch(self, action_batch: ActionBatch) -> Tuple[StateBatch, RewardBatch]:
        """
        #TODO
        """

        pass

    @staticmethod
    @abstractmethod
    def state_batch_to_graph_batch(state_batch: StateBatch) -> GraphBatch:
        """
        #TODO
        """

        pass
