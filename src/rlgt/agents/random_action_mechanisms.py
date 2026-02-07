"""
This ``Python`` module defines the `RandomActionMechanism` abstract base class, which formalizes
the concept of a random action mechanism in the context of reinforcement learning agents for graph
theory applications. A random action mechanism governs how often an agent selects a random action
instead of following its learned policy. The module also serves as a foundation for concrete random
action mechanisms that differ in how the random action probability is initialized and how it
evolves over the course of the learning process.
"""

from abc import ABC, abstractmethod


class RandomActionMechanism(ABC):
    """
    This abstract class encapsulates the concept of a random action mechanism in the context of an
    RL agent for graph theory applications. A random action mechanism controls the probability with
    which the agent selects a random action instead of an action prescribed by its policy. Concrete
    subclasses that inherit from this class must implement the following abstract property:

    * `random_action_probability`, which returns the current random action probability as a `float`
      from the interval $[0, 1]$;

    as well as the following abstract methods:

    1. `reset`, which is invoked during the agent initialization and is responsible for
       initializing the internal state of the random action mechanism; and
    2. `step`, which is invoked at the end of each iteration of the learning process and determines
       how the random action probability should be updated based on the previous best score and the
       current best score.
    """

    @abstractmethod
    def reset(self) -> None:
        """
        This abstract method must be implemented by any concrete subclass. It is invoked by an RL
        agent during the initialization process and it must initialize or reset all internal state
        maintained by the random action mechanism.
        """

        pass

    @abstractmethod
    def step(self, previous_best_score: float, current_best_score: float) -> None:
        """
        This abstract method must be implemented by any concrete subclass. It is invoked by an RL
        agent at the end of each iteration of the learning process and it must update the internal
        state of the random action mechanism based on the previous best score and the current best
        score.

        :param previous_best_score: The value of the best score before the current iteration, given
            as a `float`.
        :param current_best_score: The value of the best score after the current iteration, given
            as a `float`.
        """

        pass

    @property
    @abstractmethod
    def random_action_probability(self) -> float:
        """
        This abstract property must be implemented by any concrete subclass. It must return the
        current random action probability as a `float` value from the interval $[0, 1]$.
        """

        pass


class NoRandomActionMechanism(RandomActionMechanism):
    """
    This class inherits from the `RandomActionMechanism` class and represents a random action
    mechanism in which random actions are never executed. The random action probability is
    identically equal to 0 throughout the entire learning process.
    """

    def reset(self) -> None:
        pass

    def step(self, previous_best_score: float, current_best_score: float) -> None:
        pass

    @property
    def random_action_probability(self) -> float:
        return 0.0


class ConstantRandomActionMechanism(RandomActionMechanism):
    """
    This class inherits from the `RandomActionMechanism` class and represents a random action
    mechanism in which the probability of executing a random action is constant and specified at
    construction time.

    :ivar __random_action_probability: A `float` from the interval $[0, 1]$ that determines the
        constant probability with which a random action is executed.
    """

    def __init__(self, random_action_probability: float):
        """
        This constructor initializes the random action mechanism with a fixed random action
        probability.

        :param random_action_probability: The constant probability of executing a random action,
            given as a `float` from the interval $[0, 1]$.
        """

        self.__random_action_probability: float = random_action_probability

    def reset(self) -> None:
        pass

    def step(self, previous_best_score: float, current_best_score: float) -> None:
        pass

    @property
    def random_action_probability(self) -> float:
        return self.__random_action_probability


class ExponentialRandomActionMechanism(RandomActionMechanism):
    """
    This class inherits from the `RandomActionMechanism` class and represents a random action
    mechanism with an exponential-style adaptation rule. An initial random action probability is
    specified at construction time. If the best score does not improve for a prescribed number of
    consecutive iterations, then the random action probability is increased multiplicatively, up to
    a fixed maximum threshold. Whenever a strict improvement in the best score is observed, the
    random action probability is reset to its initial value and the adaptation process restarts.

    :ivar __initial_random_action_probability: A `float` from the interval $[0, 1]$ that represents
        the initial random action probability.
    :ivar __waiting_period: A positive `int` that specifies how many consecutive iterations without
        an improvement in the best score are required before the random action probability is
        increased.
    :ivar __multiplicative_factor: A `float` greater than 1 that specifies the factor by which the
        random action probability is multiplied when an increase is triggered.
    :ivar __maximum_random_action_probability: A `float` from the interval $[0, 1]$ that represents
        the maximum allowable value of the random action probability.
    :ivar __counter: A nonnegative `int` that counts the number of iterations since the last
        improvement in the best score or the last update of the random action probability.
    :ivar __random_action_probability: A `float` from the interval $[0, 1]$ that represents the
        current random action probability.
    """

    def __init__(
        self,
        initial_random_action_probability: float,
        waiting_period: int,
        multiplicative_factor: float,
        maximum_random_action_probability: float,
    ):
        """
        This constructor initializes the random action mechanism with the parameters governing its
        exponential adaptation behavior.

        :param initial_random_action_probability: The initial random action probability, given as a
            `float` from the interval $[0, 1]$.
        :param waiting_period: The number of consecutive iterations without an improvement in the
            best score that are required before the random action probability is increased, given
            as a positive `int`.
        :param multiplicative_factor: The multiplicative factor applied to the random action
            probability when an increase is triggered, given as a `float` greater than 1.
        :param maximum_random_action_probability: The maximum allowable value of the random action
            probability, given as a `float` from the interval $[0, 1]$.
        """

        self.__initial_random_action_probability: float = initial_random_action_probability
        self.__waiting_period: int = waiting_period
        self.__multiplicative_factor: float = multiplicative_factor
        self.__maximum_random_action_probability: float = maximum_random_action_probability

        # Set the iteration counter to 0 and the current random action probability to the initial
        # value.
        self.__counter: int = 0
        self.__random_action_probability: float = initial_random_action_probability

    def reset(self) -> None:
        # Set the iteration counter to 0 and the current random action probability to the initial
        # value.
        self.__counter = 0
        self.__random_action_probability = self.__initial_random_action_probability

    def step(self, previous_best_score: float, current_best_score: float) -> None:
        # If there is an increase in the best score, restart the counter and set the random action
        # probability to the initial value.
        if current_best_score > previous_best_score:
            self.__counter = 0
            self.__random_action_probability = self.__initial_random_action_probability

        # Otherwise, increment the counter. If the counter surpasses the waiting period, reset it
        # and multiply the current random action probability by the configured multiplicative
        # factor. Also, make sure that the current random action probability does not exceed the
        # maximum threshold value.
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
