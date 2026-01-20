"""
This ``Python`` module contains the `DeepCrossEntropyAgent` class, which encapsulates the concept
of a reinforcement learning agent to be used in graph theory applications that applies the
``PyTorch``-based deep cross entropy method.
"""

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..environments.graph_environment import EpisodeStatus, GraphEnvironment
from ..graphs.graph import Graph
from .random_action_mechanisms import RandomActionMechanism
from .graph_agent import GraphAgent


class DeepCrossEntropyAgent(GraphAgent):
    """
    This class encapsulates the concept of an RL agent to be used in graph theory applications that
    applies the ``PyTorch``-based deep cross entropy method. The agent operates over a configurable
    environment given as a `GraphEnvironment` object. In each iteration of the learning process,
    the agent generates a predetermined number of graphs through the graph-building game induced by
    the environment and computes the cumulative reward for each of these episodes run in parallel.
    The game is played by using a `torch.nn.Module` model to compute the probability for each of
    the actions to be selected in each episode and in each step. Afterwards, a certain number of
    episodes with the greatest cumulative reward are used to train the model, while another
    predetermined number of episodes with the greatest cumulative reward are carried over to the
    next generation. This finishes one iteration of the learning process. The user provides the
    model that helps select the actions to be executed. Additionally, the user can configure the
    optimizer that trains the model, given as a `torch.optim.Optimizer` object, as well as the
    corresponding loss function.

    :ivar _environment: A `GraphEnvironment` object that represents the RL environment that defines
        the extremal problem of interest and whose graph-building game should be used to construct
        all the graphs.
    :ivar _new_candidates_count: A positive `int` that determines how many graphs should be
        constructed in each iteration of the learning process by running the corresponding number
        of episodes in parallel.
    :ivar _elite_count: A positive `int` that determines how many executed episodes with the
        greatest cumulative reward should be used to train the action prediction model in each
        iteration of the learning process.
    :ivar _survivors_count: A positive `int` that determines how many executed episodes with the
        greatest cumulative reward should be carried over to the next generation in each iteration
        of the learning process.
    :ivar _policy_network: A `torch.nn.Module` object that represents the action prediction model
        that is used to compute the probability for each of the actions to be selected in each
        episode and in each step.
    :ivar _optimizer: A `torch.optim.Optimizer` object that represents the optimizer responsible
        for updating the model parameters.
    :ivar _loss_function: A function that represents the loss function used during training.
    :ivar _device: A `torch.device` object that determines on which device the action prediction
        model is located.
    :ivar _random_action_mechanism: A `RandomActionMechanism` object that indicates the random
        action mechanism that the RL agent should use.
    :ivar _step_count: A nonnegative `int` that determines the number of executed iterations of the
        learning process, if the RL agent has been initialized, and otherwise, `None`.
    :ivar _best_score: A `float` that determines the best currently achieved value for the graph
        invariant that is supposed to get maximized in the configured extremal problem, if the RL
        agent has been initialized, and otherwise, `None`.
    :ivar _population_states: If the RL agent has not been initialized, then `None`, and otherwise,
        a `numpy.ndarray` object that is used to store all the states during each iteration of the
        learning process. The shape of this tensor is ``(episode_length + 1, total_population,
        state_length)``, while its type matches that of the states from the underlying RL
        environment. Here, ``episode_length`` is the predetermined episode length from the
        underlying RL environment, ``state_length`` is the number of entries in each of the state
        vectors from the underlying RL environment, while ``total_population`` is the total number
        of executed episodes to be stored, which combines the newly generated ones with those
        carried over from the previous generation. The first dimension corresponds to the state
        trajectory from each episode, the second dimension corresponds to the executed episodes,
        while the third dimension corresponds to the vector entries. The states from the episodes
        carried over from the previous generation should appear before the states from the newly
        generated episodes.
    :ivar _population_actions: If the RL agent has not been initialized, then `None`, and
        otherwise, a `numpy.ndarray` matrix of type `numpy.int32` that is used to store all the
        actions during each iteration of the learning process. The shape of this matrix is
        ``(episode_length, total_population)``, where ``episode_length`` is the predetermined
        episode length from the underlying RL environment, while ``total_population`` is the total
        number of executed episodes to be stored, which combines the newly generated ones with
        those carried over from the previous generation. The first dimension corresponds to the
        action trajectory from each episode, while the second dimension corresponds to the executed
        episodes. The episode order must be the same as in the `_population_states` attribute.
    :ivar _population_rewards: If the RL agent has not been initialized, then `None`, and
        otherwise, a `numpy.ndarray` list of type `numpy.float32` that is used to store all the
        cumulative rewards during each iteration of the learning process. The length of this list
        is ``total_population``, i.e., the the total number of executed episodes to be stored,
        which combines the newly generated ones with those carried over from the previous
        generation. The episode order must be the same as in the `_population_states` attribute.
    """

    def __init__(
        self,
        environment: GraphEnvironment,
        new_candidates_count: int,
        elite_count: int,
        survivors_count: int,
        policy_network: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable = nn.CrossEntropyLoss(),
        random_action_mechanism: RandomActionMechanism = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        This constructor initializes an instance of the `DeepCrossEntropyAgent` class.

        :param environment: The RL environment that defines the extremal problem of interest and
            whose graph-building game should be used to construct all the graphs, given as a
            `GraphEnvironment` object.
        :param new_candidates_count: A positive `int` that determines how many graphs should be
            constructed in each iteration of the learning process by running the corresponding
            number of episodes in parallel.
        :param elite_count: A positive `int` that determines how many executed episodes with the
            greatest cumulative reward should be used to train the action prediction model in each
            iteration of the learning process.
        :param survivors_count: A positive `int` that determines how many executed episodes with
            the greatest cumulative reward should be carried over to the next generation in each
            iteration of the learning process.
        :param policy_network: The action prediction model that is used to compute the probability
            for each of the actions to be selected in each episode and in each step, given as a
            `torch.nn.Module` object.
        :param optimizer: The optimizer responsible for updating the model parameters, given as a
            `torch.optim.Optimizer` object. The parameters of the action prediction model, i.e.,
            the ``policy_network`` argument, must be passed to the optimizer.
        :param loss_function: A function that represents the loss function used during training.
            The default value is the cross entropy loss function, i.e., ``nn.CrossEntropyLoss()``.
        :param random_action_mechanism: The random action mechanism that the RL agent should use,
            given as a `RandomActionMechanism` object. If a random action should be executed, then
            it is sampled with the uniform probability distribution.
        :param rng: Either `None`, or the `numpy.random.Generator` object that represents the
            random number generator used for all the probabilistic decisions. If this argument is
            `None`, then a default `numpy.random.Generator` object will be used. The default value
            is `None`.
        """

        self._environment: GraphEnvironment = environment
        self._new_candidates_count: int = new_candidates_count
        self._elite_count: int = elite_count
        self._survivors_count: int = survivors_count

        self._policy_network: nn.Module = policy_network
        self._optimizer: torch.optim.Optimizer = optimizer
        self._loss_function: Callable = loss_function

        # Infer the device from the policy network parameters. If no parameters exist, fall back to
        # the CPU.
        params = list(self._policy_network.parameters())
        self._device: torch.device = params[0].device if params else torch.device("cpu")

        self._random_action_mechanism: RandomActionMechanism = random_action_mechanism

        # If the ``rng`` argument is `None`, then use a default `np.random.Generator`.
        if rng is None:
            rng = np.random.default_rng()
        self._rng: np.random.Generator = rng

        self._step_count: Optional[int] = None
        self._best_score: Optional[float] = None
        self._population_states: Optional[np.ndarray] = None
        self._population_actions: Optional[np.ndarray] = None
        self._population_rewards: Optional[np.ndarray] = None

    def reset(self) -> None:
        # Initialize the step count to 0 and the best score to minus infinity.
        self._step_count = 0
        self._best_score = float("-inf")

        # Initialize the population states, the population actions and the population rewards to
        # the zero `np.ndarray` objects of the required shape and type.
        total_population = self._survivors_count + self._new_candidates_count
        self._population_states = np.zeros(
            (
                self._environment.episode_length + 1,
                total_population,
                self._environment.state_length,
            ),
            dtype=self._environment.state_dtype,
        )
        self._population_actions = np.zeros(
            (self._environment.episode_length, total_population), dtype=np.int32
        )
        self._population_rewards = np.zeros((total_population,), dtype=np.float32)

    def step(self) -> None:
        # Initialize a batch of episodes with the batch size ``_new_candidates_count`` and store
        # the starting states to the ``_population_states`` attribute. While storing the states,
        # the final ``_new_candidates_count`` positions should be used (in the second dimension),
        # while the starting ``_survivors_count`` positions are reserved for the surviving episodes
        # carried over from the previous generation.
        state_batch, status = self._environment.reset_batch(batch_size=self._new_candidates_count)
        self._population_states[0, self._survivors_count :, :] = state_batch
        self._population_rewards[self._survivors_count :] = 0

        # Set the episode action counter to 0 and use the random action mechanism to obtain the
        # random action probability.
        episode_action_count = 0
        random_action_probability = self._random_action_mechanism.random_action_probability

        # While the episodes are in progress...
        while status == EpisodeStatus.IN_PROGRESS:
            # Use the policy network to get the probability distribution for each action to be
            # selected for execution in each of the parallelized episodes.
            state_batch_torch = torch.from_numpy(state_batch.astype(np.float32)).to(self._device)
            logits_batch_torch = self._policy_network(state_batch_torch)

            # Make it impossible to execute an action that is not available for execution.
            action_mask = self._environment.action_mask
            if action_mask is not None:
                action_mask_torch = torch.from_numpy(action_mask).to(self._device)
                logits_batch_torch = logits_batch_torch.masked_fill(
                    ~action_mask_torch, float("-inf")
                )

            action_batch_torch = Categorical(logits=logits_batch_torch).sample()
            action_batch = action_batch_torch.cpu().numpy()

            random_mask = (
                self._rng.random(size=(action_batch.shape[0],)) < random_action_probability
            )
            entry_count = np.count_nonzero(random_mask)
            action_batch[random_mask] = self._rng.integers(
                low=0, high=self._environment.action_number, size=entry_count, dtype=np.int32
            )

            self._population_actions[episode_action_count, self._survivors_count :] = action_batch
            state_batch, reward_batch, status = self._environment.step_batch(action_batch)

            self._population_rewards[self._survivors_count :] += reward_batch
            episode_action_count += 1
            self._population_states[episode_action_count, self._survivors_count :, :] = state_batch

        elite_mask = np.zeros((self._survivors_count + self._new_candidates_count), dtype=bool)
        survivors_mask = np.zeros((self._survivors_count + self._new_candidates_count), dtype=bool)

        if self._step_count == 0:
            elite_mask[
                self._survivors_count
                + np.argpartition(
                    self._population_rewards[self._survivors_count :], -self._elite_count
                )[-self._elite_count :]
            ] = True
            survivors_mask[
                self._survivors_count
                + np.argpartition(
                    self._population_rewards[self._survivors_count :], -self._survivors_count
                )[-self._survivors_count :]
            ] = True
        else:
            elite_mask[
                np.argpartition(self._population_rewards, -self._elite_count)[-self._elite_count :]
            ] = True
            survivors_mask[
                np.argpartition(self._population_rewards, -self._survivors_count)[
                    -self._survivors_count :
                ]
            ] = True

        elite_states = self._population_states[:-1, elite_mask, :].reshape(
            -1, self._environment.state_length
        )
        elite_actions = self._population_actions[:, elite_mask].reshape(-1)
        elite_states_torch = torch.from_numpy(elite_states.astype(np.float32)).to(self._device)
        elite_actions_torch = torch.from_numpy(elite_actions.astype(np.int64)).to(self._device)

        self._optimizer.zero_grad()
        logits_torch = self._policy_network(elite_states_torch)
        loss = self._loss_function(logits_torch, elite_actions_torch)
        loss.backward()
        self._optimizer.step()

        if self._step_count == 0:
            assert np.all(survivors_mask[: self._survivors_count] == False)

        self._population_states[:, : self._survivors_count, :] = self._population_states[
            :, survivors_mask, :
        ]
        self._population_actions[:, : self._survivors_count] = self._population_actions[
            :, survivors_mask
        ]
        self._population_rewards[: self._survivors_count] = self._population_rewards[
            survivors_mask
        ]

        new_best_score = np.max(self._population_rewards[: self._survivors_count]).item()
        if self._best_score is not None:
            self._is_best_score_improved = not np.isclose(self._best_score, new_best_score)
        else:
            self._is_best_score_improved = True
        self._best_score = new_best_score

        self._step_count += 1

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def best_score(self) -> Optional[float]:
        return self._best_score

    @property
    def best_graph(self) -> Optional[Graph]:
        if self._best_score is None:
            return None

        best_index = np.argmax(self._population_rewards[: self._survivors_count])
        best_state = self._population_states[self._environment.episode_length, best_index, :]
        best_graph = self._environment.state_to_graph(best_state)

        return best_graph
