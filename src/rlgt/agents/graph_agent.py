"""
This ``Python`` module defines the `GraphAgent` abstract base class, which formalizes the concept
of a reinforcement learning agent for graph theory applications.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..graphs.graph import Graph


class GraphAgent(ABC):
    """
    This abstract class encapsulates the concept of a reinforcement learning agent for graph theory
    applications. Such an agent is intended to solve extremal problems in which a given graph
    invariant is to be maximized over a finite family of fully colored *k*-edge-colored looped
    complete graphs. The agent interacts iteratively with a reinforcement learning environment. The
    environment encodes the extremal problem, including the state representation, the available
    actions and the transition dynamics, while the agent is responsible for steering the learning
    process through successive decisions.

    Any concrete subclass must implement the following abstract methods:

    1. `reset`, which initializes or re-initializes the agent and prepares it to start the learning
       process; and
    2. `step`, which performs a single iteration of the learning process.

    In addition, any concrete subclass must implement the following abstract properties:

    1. `step_count`, which returns the number of executed learning iterations;
    2. `best_score`, which returns the best value of the target graph invariant achieved so far;
       and
    3. `best_graph`, which returns a graph attaining that best value.
    """

    @abstractmethod
    def reset(self) -> None:
        """
        This abstract method must be implemented by any concrete subclass. It must initialize the
        agent and prepare it to begin the learning process. If the agent has been used previously,
        invoking this method must reset all internal state so that the learning restarts from
        scratch.
        """

        pass

    @abstractmethod
    def step(self) -> None:
        """
        This abstract method must be implemented by any concrete subclass. It must perform a single
        iteration of the learning process, which may involve one or more interactions between the
        agent and the environment. This iteration should update the agent's internal state and
        improve its policy or decision-making strategy based on the observed outcomes.
        """

        pass

    @property
    @abstractmethod
    def step_count(self) -> Optional[int]:
        """
        This abstract property must be implemented by any concrete subclass. It must return the
        number of learning iterations executed so far. If the agent has been initialized, the
        returned value must be a nonnegative `int`. If the agent has not yet been initialized, the
        value `None` must be returned.
        """

        pass

    @property
    @abstractmethod
    def best_score(self) -> Optional[float]:
        """
        This abstract property must be implemented by any concrete subclass. It must return the
        best value of the target graph invariant achieved so far. If at least one learning
        iteration has been executed, the value is returned as a `float`. If the agent has been
        initialized but no iterations have yet been executed, the value −∞ must be returned. If the
        agent has not been initialized, the value `None` must be returned.
        """

        pass

    @property
    @abstractmethod
    def best_graph(self) -> Optional[Graph]:
        """
        This abstract property must be implemented by any concrete subclass. It must return a graph
        attaining the best value of the target graph invariant achieved so far. If at least one
        learning iteration has been executed, the result must be returned as a `Graph` object.
        Otherwise, if no iterations have been executed or the agent has not been initialized, the
        value `None` must be returned.
        """

        pass
