from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np
from graphs.graph import GraphBatch


class StateBatch:
    """
    #TODO
    """

    def __init__(self, state_batch: np.ndarray):
        self.__state_batch: np.ndarray = state_batch

    @property
    def data(self) -> np.ndarray:
        """
        #TODO
        """

        return self.__state_batch


class ActionBatch:
    """
    #TODO
    """

    def __init__(self, action_batch: np.ndarray):
        self.__action_batch: np.ndarray = action_batch

    @property
    def data(self) -> np.ndarray:
        """
        #TODO
        """

        return self.__action_batch


class RewardBatch:
    """
    #TODO
    """

    def __init__(self, reward_batch: np.ndarray):
        self.__reward_batch: np.ndarray = reward_batch

    @property
    def data(self) -> np.ndarray:
        """
        #TODO
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
