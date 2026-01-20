"""
This ``Python`` module contains the `GraphAgent` class, which encapsulates the concept of a
reinforcement learning agent to be used in graph theory applications.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..graphs.graph import Graph


class GraphAgent(ABC):
    """
    This class encapsulates the concept of an RL agent to be used in graph theory applications. The
    goal of such an agent is to tackle extremal problems where a given graph invariant should be
    maximized over some finite set of fully colored $k$-edge-colored looped complete graphs. Here,
    the agent guides the learning process by iteratively performing interactions on a provided RL
    environment, while the environment contains all the extremal problem related information and
    all the information concerning the states, actions and transitions.

    The concrete classes that inherit from this abstract class must implement the following two
    abstract methods:

    1. `reset`, which serves to initialize an RL agent and prepare it to start the learning
       process; and
    2. `step`, which performs one iteration of the learning process;

    as well as the following three abstract properties:

    1. `step_count`, which returns the number of executed iterations of the learning process;
    2. `best_score`, which returns the best currently achieved value for the graph invariant that
       is supposed to get maximized in the configured extremal problem; and
    3. `best_graph`, which returns a graph that attains the best currently achieved value for the
       graph invariant that is supposed to get maximized in the configured extremal problem.
    """

    @abstractmethod
    def reset(self) -> None:
        """
        This abstract method must be implemented in any concrete class that inherits from the
        `GraphAgent` class. It should initialize an RL agent and prepare it to start the learning
        process. If the agent has already been initialized and potentially used, then this method
        should re-initialize the agent and restart the learning process.
        """

        pass

    @abstractmethod
    def step(self) -> None:
        """
        This abstract method must be implemented in any concrete class that inherits from the
        `GraphAgent` class. It should perform one iteration of the learning process.
        """

        pass

    @property
    @abstractmethod
    def step_count(self) -> int:
        """
        This abstract property must be implemented in any concrete class that inherits from the
        `GraphAgent` class. It should return the number of executed iterations of the learning
        process, as a nonnegative `int`.
        """

        pass

    @property
    @abstractmethod
    def best_score(self) -> float:
        """
        This abstract property must be implemented in any concrete class that inherits from the
        `GraphAgent` class. If at least one iteration of the learning process has been executed,
        the property should return the best currently achieved value for the graph invariant that
        is supposed to get maximized in the configured extremal problem, as a `float`. Otherwise,
        if no iterations of the learning process have been executed yet, then the value −∞ should
        be returned, again as a `float`.
        """

        pass

    @property
    @abstractmethod
    def best_graph(self) -> Optional[Graph]:
        """
        This abstract property must be implemented in any concrete class that inherits from the
        `GraphAgent` class. If at least one iteration of the learning process has been executed,
        the property should return a graph that attains the best currently achieved value for the
        graph invariant that is supposed to get maximized in the configured extremal problem, as a
        `Graph` object. Otherwise, if no iterations of the learning process have been executed yet,
        then `None` should be returned.
        """

        pass
