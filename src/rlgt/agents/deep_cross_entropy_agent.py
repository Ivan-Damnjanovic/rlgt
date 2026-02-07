"""
This ``Python`` module contains the `DeepCrossEntropyAgent` class, which implements a reinforcement
learning agent for graph theory applications using the Deep Cross-Entropy method with ``PyTorch``.
"""

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..environments.graph_environment import EpisodeStatus, GraphEnvironment
from ..graphs.graph import Graph
from .graph_agent import GraphAgent
from .random_action_mechanisms import NoRandomActionMechanism, RandomActionMechanism


class DeepCrossEntropyAgent(GraphAgent):
    """
    This class encapsulates a reinforcement learning agent for graph theory applications using the
    ``PyTorch``-based Deep Cross-Entropy method. The agent operates on a configurable environment
    given as a `GraphEnvironment` object. In each iteration of the learning process, the agent
    generates a predetermined number of graphs through the graph building game defined by the
    environment and computes the graph invariant value for the final underlying graph in each
    episode run in parallel. The agent uses a `torch.nn.Module` model to compute the probability of
    selecting each action at each step of every episode. A configured number of episodes with the
    highest graph invariant value are used to train the model, while another configured number of
    top episodes are carried over to the next generation. This completes one iteration of the
    learning process. The user provides the model, the optimizer for training the model with
    cross-entropy loss, and a random action mechanism to guide exploration. When a random action
    occurs, it is selected uniformly among all actions available in the current state.

    :ivar _environment: A `GraphEnvironment` object defining the extremal problem and providing the
        graph building game used to construct all the graphs.
    :ivar _policy_network: A `torch.nn.Module` object predicting the action probabilities for each
        step in each episode.
    :ivar _optimizer: A `torch.optim.Optimizer` object that updates the model parameters.
    :ivar _loss_function: A function implementing the cross-entropy loss used for training.
    :ivar _device: A `torch.device` object indicating the device where the model resides.
    :ivar _candidates_count: A positive `int` specifying the number of graphs constructed per
        iteration, i.e., the number of episodes run in parallel.
    :ivar _elite_count: A positive `int` specifying the number of top-performing episodes used to
        train the model in each iteration.
    :ivar _survivors_count: A positive `int` specifying the number of top-performing episodes
        carried over to the next generation.
    :ivar _random_action_mechanism: A `RandomActionMechanism` object that determines the
        probability of executing a random action. When a random action is selected, it is sampled
        uniformly among all available actions in the current state.
    :ivar _random_generator: A `numpy.random.Generator` object used for all probabilistic
        decisions.
    :ivar _step_count: A nonnegative `int` representing the number of executed iterations, or
        `None` if the agent has not been initialized.
    :ivar _best_score: A `float` representing the best achieved value for the graph invariant, or
        `None` if the agent has not been initialized.
    :ivar _population_states: Either `None` if uninitialized, or a `numpy.ndarray` storing all
        states during each iteration. Its shape is ``(episode_length + 1, total_population,
        state_length)``, where ``episode_length`` is the episode length of the RL environment,
        ``state_length`` is the length of the state vector, and ``total_population`` is the total
        number of episodes stored. The stored episodes include both newly generated episodes and
        those carried over from the previous generation. The first dimension corresponds to the
        state trajectory of each episode, the second to the executed episodes, and the third to the
        state vector entries. The states from carried-over episodes appear before the newly
        generated ones.
    :ivar _population_actions: Either `None` if uninitialized, or a `numpy.ndarray` of type
        `numpy.int32` storing all actions during each iteration. Its shape is ``(episode_length,
        total_population)`` with the first dimension as the action trajectory and the second as the
        executed episodes. The episode order matches `_population_states`.
    :ivar _population_scores: Either `None` if uninitialized, or a `numpy.ndarray` of type
        `numpy.float32` storing the graph invariant value for each episode. Its length is
        ``total_population`` and the episode order matches `_population_states`.
    """

    def __init__(
        self,
        environment: GraphEnvironment,
        policy_network: nn.Module,
        optimizer: torch.optim.Optimizer,
        candidates_count: int = 200,
        elite_count: int = 30,
        survivors_count: int = 50,
        random_action_mechanism: RandomActionMechanism = NoRandomActionMechanism(),
        random_generator: Optional[np.random.Generator] = None,
    ):
        """
        This constructor initializes an instance of the `DeepCrossEntropyAgent` class.

        :param environment: The RL environment defining the extremal problem and providing the
            graph building game, given as a `GraphEnvironment` object.
        :param policy_network: The policy network used to compute the probability of each action in
            each episode and step, given as a `torch.nn.Module` object.
        :param optimizer: The optimizer responsible for updating the model parameters, given as a
            `torch.optim.Optimizer` object. The parameters of `policy_network` must be passed to
            it.
        :param candidates_count: A positive `int` specifying how many graphs are generated in each
            iteration by running the corresponding number of episodes in parallel. The default
            value is 200.
        :param elite_count: A positive `int` specifying how many episodes with the greatest graph
            invariant value are used to train the model in each iteration. The default value is 30.
        :param survivors_count: A positive `int` specifying how many episodes with the greatest
            graph invariant value are carried over to the next generation in each iteration. The
            default value is 50.
        :param random_action_mechanism: A `RandomActionMechanism` object that governs the
            probability of executing a random action in each step of the graph building game. When
            a random action is triggered, the agent ignores the action predicted by the policy
            network and instead selects an action uniformly at random among all available actions.
            By default, this is ``NoRandomActionMechanism()``, meaning that no random actions are
            ever executed.
        :param random_generator: Either `None`, or a `numpy.random.Generator` used for
            probabilistic decisions. If `None`, a default generator will be used. The default value
            is `None`.
        """

        self._environment: GraphEnvironment = environment
        # Enforce the sparse setting.
        self._environment.sparse_setting = True

        self._policy_network: nn.Module = policy_network
        self._optimizer: torch.optim.Optimizer = optimizer
        self._loss_function: Callable = nn.CrossEntropyLoss()

        # Infer the device from the policy network parameters. If no parameters exist, fall back to
        # the CPU.
        params = list(self._policy_network.parameters())
        self._device: torch.device = params[0].device if params else torch.device("cpu")

        self._candidates_count: int = candidates_count
        self._elite_count: int = elite_count
        self._survivors_count: int = survivors_count
        self._random_action_mechanism: RandomActionMechanism = random_action_mechanism

        # If the ``random_generator`` argument is `None`, then use a default `np.random.Generator`.
        if random_generator is None:
            random_generator = np.random.default_rng()
        self._random_generator: np.random.Generator = random_generator

        self._step_count: Optional[int] = None
        self._best_score: Optional[float] = None
        self._population_states: Optional[np.ndarray] = None
        self._population_actions: Optional[np.ndarray] = None
        self._population_scores: Optional[np.ndarray] = None

    def reset(self) -> None:
        # Initialize the step count to 0 and the best score to minus infinity. Also, initialize the
        # random action mechanism.
        self._step_count = 0
        self._best_score = float("-inf")
        self._random_action_mechanism.reset()

        # Initialize the population states, the population actions and the population scores to the
        # zero `np.ndarray` objects of the required shape and type.
        total_population = self._survivors_count + self._candidates_count
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
        self._population_scores = np.zeros((total_population,), dtype=np.float32)

    def step(self) -> None:
        # Initialize a batch of episodes with the batch size ``_candidates_count`` and store the
        # starting states to the ``_population_states`` attribute. While storing the states, the
        # final ``_candidates_count`` positions should be used (in the second dimension), while the
        # starting ``_survivors_count`` positions are reserved for the surviving episodes carried
        # over from the previous generation.
        state_batch, _, status = self._environment.reset_batch(batch_size=self._candidates_count)
        self._population_states[0, self._survivors_count :, :] = state_batch

        # Set the episode action counter to 0 and use the random action mechanism to obtain the
        # random action probability.
        episode_action_count = 0
        random_action_probability = self._random_action_mechanism.random_action_probability

        # While the episodes are in progress...
        while status == EpisodeStatus.IN_PROGRESS:
            # Use the policy network to get the probability distribution for each action to be
            # selected for execution in each of the episodes run in parallel.
            state_batch_torch = torch.from_numpy(state_batch.astype(np.float32)).to(self._device)
            logits_batch_torch = self._policy_network(state_batch_torch)

            # Make it impossible to execute an action that is not available for execution.
            action_mask = self._environment.action_mask
            if action_mask is not None:
                action_mask_torch = torch.from_numpy(action_mask).to(self._device)
                logits_batch_torch = logits_batch_torch.masked_fill(
                    ~action_mask_torch, float("-inf")
                )

            # Sample the actions according to the obtained probability distributions.
            action_batch_torch = Categorical(logits=logits_batch_torch).sample()
            action_batch = action_batch_torch.cpu().numpy()

            # Use the random action probability to decide whether each sampled action should be
            # replaced by a random action.
            random_mask = (
                self._random_generator.random(size=action_batch.shape[0])
                < random_action_probability
            )

            # Select each required random action among the actions available for execution using
            # the uniform probability distribution.
            if np.any(random_mask):
                # Settle the case where at least one action is not available for execution.
                if action_mask is not None:
                    probabilities_batch = action_mask[random_mask].astype(np.float32)
                    probabilities_batch /= probabilities_batch.sum(axis=1, keepdims=True)

                    action_batch[random_mask] = np.array(
                        [
                            self._random_generator.choice(
                                probabilities_batch.shape[1], p=probabilities
                            )
                            for probabilities in probabilities_batch
                        ],
                        dtype=np.int32,
                    )

                # Settle the case where all the actions are available for execution.
                else:
                    entry_count = np.count_nonzero(random_mask)
                    action_batch[random_mask] = self._random_generator.integers(
                        low=0,
                        high=self._environment.action_number,
                        size=entry_count,
                        dtype=np.int32,
                    )

            # Store the selected actions and execute them.
            self._population_actions[episode_action_count, self._survivors_count :] = action_batch
            state_batch, graph_invariant_batch, status = self._environment.step_batch(action_batch)

            # Store the newly obtained states.
            episode_action_count += 1
            self._population_states[episode_action_count, self._survivors_count :, :] = state_batch

        # Store the graph invariant values, i.e., the scores.
        self._population_scores[self._survivors_count :] = graph_invariant_batch

        # Initialize the mask that decides which executed episodes should be used to train the
        # action prediction model.
        elite_mask = np.zeros((self._survivors_count + self._candidates_count), dtype=bool)
        # Initialize the mask that decides which executed episodes should be carried over to the
        # next generation.
        survivors_mask = np.zeros((self._survivors_count + self._candidates_count), dtype=bool)

        # In the first iteration of the learning process, only the last `_candidates_count`
        # executed episodes should be considered because there are no survivor episodes. Therefore,
        # the first `_survivors_count` episodes correspond to nonexisting episodes, hence they
        # should be ignored.
        if self._step_count == 0:
            elite_mask[
                self._survivors_count
                + np.argpartition(
                    self._population_scores[self._survivors_count :], -self._elite_count
                )[-self._elite_count :]
            ] = True
            survivors_mask[
                self._survivors_count
                + np.argpartition(
                    self._population_scores[self._survivors_count :], -self._survivors_count
                )[-self._survivors_count :]
            ] = True
        # In any subsequent iteration of the learning process, there are survivor episodes from the
        # previous iteration, so all the episodes from the `np.ndarray` are valid, which means that
        # they should all be considered.
        else:
            elite_mask[
                np.argpartition(self._population_scores, -self._elite_count)[-self._elite_count :]
            ] = True
            survivors_mask[
                np.argpartition(self._population_scores, -self._survivors_count)[
                    -self._survivors_count :
                ]
            ] = True

        # Extract the (state, action) pairs from the elite executed episodes, i.e., the executed
        # episodes that should be used to train the policy network.
        elite_states = self._population_states[:-1, elite_mask, :].reshape(
            -1, self._environment.state_length
        )
        elite_actions = self._population_actions[:, elite_mask].reshape(-1)
        elite_states_torch = torch.from_numpy(elite_states.astype(np.float32)).to(self._device)
        elite_actions_torch = torch.from_numpy(elite_actions.astype(np.int64)).to(self._device)

        # Use the optimizer to train the policy network.
        self._optimizer.zero_grad()
        logits_torch = self._policy_network(elite_states_torch)
        loss = self._loss_function(logits_torch, elite_actions_torch)
        loss.backward()
        self._optimizer.step()

        # Extract the states, actions and scores from the survivor episodes and store them so that
        # they are available in the next generation.
        self._population_states[:, : self._survivors_count, :] = self._population_states[
            :, survivors_mask, :
        ]
        self._population_actions[:, : self._survivors_count] = self._population_actions[
            :, survivors_mask
        ]
        self._population_scores[: self._survivors_count] = self._population_scores[survivors_mask]

        # Update the best score, and the random action probability through the random action
        # mechanism.
        current_best_score = np.max(self._population_scores[: self._survivors_count]).item()
        self._random_action_mechanism.step(
            previous_best_score=self._best_score, current_best_score=current_best_score
        )
        self._best_score = current_best_score

        # Increment the number of executed iterations of the learning process.
        self._step_count += 1

    @property
    def step_count(self) -> Optional[int]:
        return self._step_count

    @property
    def best_score(self) -> Optional[float]:
        return self._best_score

    @property
    def best_graph(self) -> Optional[Graph]:
        if self._step_count is None or self._step_count < 1:
            return None

        best_index = np.argmax(self._population_scores[: self._survivors_count])
        best_state = self._population_states[self._environment.episode_length, best_index, :]
        best_graph = self._environment.state_to_graph(best_state)

        return best_graph
