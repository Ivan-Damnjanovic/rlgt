"""
This ``Python`` module contains the `RandomActionMechanism` abstract class, which encapsulates the
concept of a random action mechanism in the context of a reinforcement learning agent to be used
in graph theory applications, alongside several concrete classes that inherit from this abstract
class.
"""


from abc import ABC, abstractmethod
from typing import Callable


class RandomActionMechanism(ABC):
    """
    This abstract class encapsulates the concept of a random action mechanism in the context of an
    RL agent to be used in graph theory applications. The instances of this class are callable
    objects that output the probability of a random action being executed in the following step of
    an RL agent. In this case, the input is just a `bool` that indicates whether the current best
    score is a strict improvement over the previous best score from one iteration ago. Therefore,
    the concrete classes that inherit from this class must implement the abstract method `__call__`
    that accepts a  `bool` and returns a `float` from the interval $[0, 1]$.
    """

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def step(self, previous_best_score: float, new_best_score: float) -> None:
        pass

    @property
    @abstractmethod
    def random_action_probability(self) -> float:
        pass

    # @abstractmethod
    # def __call__(self, has_best_score_improved: bool) -> float:
    #     """
    #     This abstract method must be implemented in any concrete class that inherits from the
    #     `RandomActionMechanism` class. It should accept a `bool` that indicates whether the current
    #     best score is a strict improvement over the previous best score, and return a `float` from
    #     the interval $[0, 1]$ that determines the probability of a random action being executed in
    #     the following step.

    #     :param has_best_score_improved: A `bool` that indicates whether the current best score is a
    #         strict improvement over the previous best score from one iteration ago.
        
    #     :return: A `float` from the interval $[0, 1]$ that determines the probability of a random
    #         action being executed in the following step.
    #     """

    #     pass


class NoRandomActionMechanism(RandomActionMechanism):
    """
    This class inherits from the `RandomActionMechanism` class and it is used to instantiate random
    action mechanisms where there is no random action, i.e., where the probability of executing a
    random action is always equal to 0.
    """

    def reset(self) -> None:
        pass
    
    def step(self, previous_best_score: float, new_best_score: float) -> None:
        pass

    @property
    def random_action_probability(self) -> float:
        return 0.0


class ConstantRandomActionMechanism(RandomActionMechanism):
    """
    This class inherits from the `RandomActionMechanism` class and it is used to instantiate random
    action mechanisms where the probability of executing a random action is constant, and it is
    equal to a provided probability.

    :ivar __random_action_probability: A `float` from $[0, 1]$ that determines the constant
        probability for a random action to be executed.
    """

    def __init__(self, random_action_probability: float):
        """
        This constructor initializes the desired random action mechanism with a constant random
        action probability.

        :param random_action_probability: The provided constant random action probability, given as
            `float` from the interval $[0, 1]$.
        """

        self.__random_action_probability: float = random_action_probability
    
    def reset(self) -> None:
        pass
    
    def step(self, previous_best_score: float, new_best_score: float) -> None:
        pass
    
    @property
    def random_action_probability(self) -> float:
        return self.__random_action_probability


class ExponentialRandomActionMechanism(RandomActionMechanism):
    """
    This class inherits from the `RandomActionMechanism` class and it is used to instantiate random
    action mechanisms such that there is an initial random action probability, and after each
    predetermined number of steps without an improvement in the best score, the random action
    probability increases by a configured multiplicative factor, up to a maximum threshold value.
    As soon as the best score is strictly improved, the random action probability is reset to the
    initial value and the entire procedure starts over again.

    :ivar __initial_random_action_probability: A `float` from $[0, 1]$ that represents the initial
        random action probability.
    :ivar __waiting_period: A positive `int` that determines how many steps without an improvement
        in the best score are needed for the random action probability to get increased.
    :ivar __multiplicative_factor: A `float` greater than 1 that determines the multiplicative
        factor by which the random action probability should get increased when needed.
    :ivar __maximum_random_action_probability: A `float` from $[0, 1]$ that represents the maximum
        threshold value for the random action probability.
    """

    def __init__(
        self,
        initial_random_action_probability: float,
        waiting_period: int,
        multiplicative_factor: float,
        maximum_random_action_probability: float
    ):
        """
        This constructor initializes the desired random action mechanism with an exponential-like
        behavior.

        :param initial_random_action_probability: The initial random action probability, given as a
            `float` from the interval $[0, 1]$.
        :param waiting_period: The number of steps without an improvement in the best score that
            are needed for the random action probability to get increased, given as a positive
            `int`.
        :param multiplicative_factor: The multiplicative factor by which the random action
            probability should get increased when needed, given as a `float` greater than 1.
        :param maximum_random_action_probability: The maximum threshold value for the random action
            probability, given as a `float` from the interval $[0, 1]$. 
        """

        self.__initial_random_action_probability: float = initial_random_action_probability
        self.__waiting_period: int = waiting_period
        self.__multiplicative_factor: float = multiplicative_factor
        self.__maximum_random_action_probability: float = maximum_random_action_probability

        self.__counter: int = 0
        self.__random_action_probability: float = initial_random_action_probability

    def reset(self) -> None:
        self.__counter = 0
        self.__random_action_probability = initial_random_action_probability

    def step(self, previous_best_score: float, new_best_score: float) -> None:
        if new_best_score > previous_best_score:
            self.__counter = 0
            self.__random_action_probability = initial_random_action_probability

        else:
            self.__counter += 1

            if self.__counter >= self.__waiting_period:
                self.__counter -= self.__waiting_period
                self.__random_action_probability *= self.__multiplicative_factor
                self.__random_action_probability = min(
                    self.__random_action_probability, self.__maximum_random_action_probability
                )
    
    @property
    def random_action_probability(self) -> float:
        return self.__random_action_probability

