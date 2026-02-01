"""
This ``Python`` module contains the `ReinforceAgent` class, which encapsulates the concept of a
reinforcement learning agent to be used in graph theory applications that applies the
``PyTorch``-based REINFORCE method.
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


class ReinforceAgent(GraphAgent):
    """
    This class encapsulates the concept of an RL agent to be used in graph theory applications that
    applies the ``PyTorch``-based REINFORCE algorithm. The agent operates over a configurable
    environment given as a `GraphEnvironment` object. In each iteration of the learning process,
    the agent generates a predetermined number of graphs through the graph-building game induced by
    the environment and computes the cumulative reward for each of these episodes. The game is played
    by using a policy network (`torch.nn.Module`) that computes action probabilities. The agent uses
    the REINFORCE algorithm (policy gradient with Monte Carlo returns) to update the network based on
    collected experiences.

    :ivar _environment: A `GraphEnvironment` object that represents the RL environment that defines
        the extremal problem of interest and whose graph-building game should be used to construct
        all the graphs.
    :ivar _policy_network: A `torch.nn.Module` object that represents the policy network
        that is used to compute the probability for each of the actions to be selected.
    :ivar _optimizer: A `torch.optim.Optimizer` object that represents the optimizer responsible
        for updating network parameters.
    :ivar _device: A `torch.device` object that determines on which device the network is located.
    :ivar _batch_size: A positive `int` that determines how many episodes should be collected before
        performing a policy update.
    :ivar _gamma: A `float` in the range [0, 1] representing the discount factor for returns.
    :ivar _use_baseline: A `bool` indicating whether to use a baseline (average return) for variance reduction.
    :ivar _elite_fraction: A `float` in the range (0, 1] determining what fraction of episodes to use for training.
        If 1.0, all episodes are used. If less, only the top elite_fraction episodes are used (like Cross Entropy).
    :ivar _random_action_mechanism: A `RandomActionMechanism` object that indicates the random
        action mechanism that the RL agent should use.
    :ivar _rng: The `numpy.random.Generator` object that represents the random number generator
        used for all the probabilistic decisions.
    :ivar _step_count: A nonnegative `int` that determines the number of executed iterations of the
        learning process, if the RL agent has been initialized, and otherwise, `None`.
    :ivar _best_score: A `float` that determines the best currently achieved value for the graph
        invariant that is supposed to get maximized in the configured extremal problem, if the RL
        agent has been initialized, and otherwise, `None`.
    """

    def __init__(
        self,
        environment: GraphEnvironment,
        new_candidates_count: int,
        elite_count: int,
        survivors_count: int,
        discount_factor: float,
        use_baseline: bool,
        policy_network: nn.Module,
        optimizer: torch.optim.Optimizer,
        random_action_mechanism: RandomActionMechanism = NoRandomActionMechanism(),
        rng: Optional[np.random.Generator] = None,
    ):
        """
        This constructor initializes an instance of the `ReinforceAgent` class.

        :param environment: The RL environment that defines the extremal problem of interest and
            whose graph-building game should be used to construct all the graphs, given as a
            `GraphEnvironment` object.
        :param policy_network: The policy network that is used to compute the probability
            for each of the actions to be selected, given as a `torch.nn.Module` object.
        :param optimizer: The optimizer responsible for updating network parameters, given as a
            `torch.optim.Optimizer` object. The parameters of the policy network must be passed
            to the optimizer.
        :param batch_size: A positive `int` that determines how many episodes should be collected
            before performing a policy update. The default value is 32.
        :param gamma: A `float` in the range [0, 1] representing the discount factor for returns.
            The default value is 0.99.
        :param use_baseline: A `bool` indicating whether to use a baseline (average return) for
            variance reduction. The default value is True.
        :param elite_fraction: A `float` in the range (0, 1] determining what fraction of episodes
            to use for training. If 1.0, all episodes are used. If less (e.g., 0.1), only the top
            elite_fraction episodes are used for training (like Cross Entropy Method). The default
            value is 1.0.
        :param random_action_mechanism: The random action mechanism that the RL agent should use,
            given as a `RandomActionMechanism` object. The default value is
            ``NoRandomActionMechanism()``, i.e., no random actions should be executed by default.
        :param rng: Either `None`, or the `numpy.random.Generator` object that represents the
            random number generator used for all the probabilistic decisions. If this argument is
            `None`, then a default `numpy.random.Generator` object will be used. The default value
            is `None`.
        """

        self._environment: GraphEnvironment = environment
        self._new_candidates_count: int = new_candidates_count
        self._elite_count: int = elite_count
        self._survivors_count: int = survivors_count
        self._discount_factor: float = discount_factor
        self._use_baseline: bool = use_baseline

        self._policy_network: nn.Module = policy_network
        self._optimizer: torch.optim.Optimizer = optimizer

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
        self._population_returns: Optional[np.ndarray] = None

    def reset(self) -> None:
        # Initialize the step count to 0 and the best score to minus infinity. Also, initialize the
        # random action mechanism.
        self._step_count = 0
        self._best_score = float("-inf")
        self._random_action_mechanism.reset()

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
        self._population_returns = np.zeros(
            (self._environment.episode_length, total_population), dtype=np.float32
        )

    def step(self) -> None:
        # Initialize a batch of episodes with the batch size ``_new_candidates_count`` and store
        # the starting states to the ``_population_states`` attribute. While storing the states,
        # the final ``_new_candidates_count`` positions should be used (in the second dimension),
        # while the starting ``_survivors_count`` positions are reserved for the surviving episodes
        # carried over from the previous generation.
        state_batch, status = self._environment.reset_batch(batch_size=self._new_candidates_count)
        self._population_states[0, self._survivors_count :, :] = state_batch
        self._population_rewards[self._survivors_count :] = 0
        self._population_returns[:, self._survivors_count :] = 0

        # Set the episode action counter to 0 and use the random action mechanism to obtain the
        # random action probability.
        episode_action_count = 0
        random_action_probability = self._random_action_mechanism.random_action_probability

        while status == EpisodeStatus.IN_PROGRESS:
            # Select actions for all active episodes
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
            random_mask = self._rng.random(size=action_batch.shape[0]) < random_action_probability

            # Select each required random action among the actions available for execution using
            # the uniform probability distribution.
            if np.any(random_mask):
                # Settle the case where at least one action is not available for execution.
                if action_mask is not None:
                    probabilities_batch = action_mask[random_mask].astype(np.float32)
                    probabilities_batch /= probabilities_batch.sum(axis=1, keepdims=True)

                    action_batch[random_mask] = np.array(
                        [
                            self._rng.choice(probabilities_batch.shape[1], p=probabilities)
                            for probabilities in probabilities_batch
                        ],
                        dtype=np.int32,
                    )

                # Settle the case where all the actions are available for execution.
                else:
                    entry_count = np.count_nonzero(random_mask)
                    action_batch[random_mask] = self._rng.integers(
                        low=0,
                        high=self._environment.action_number,
                        size=entry_count,
                        dtype=np.int32,
                    )

            # Store the selected actions and execute them.
            self._population_actions[episode_action_count, self._survivors_count :] = action_batch
            state_batch, reward_batch, status = self._environment.step_batch(action_batch)

            # Update the cumulative rewards and store the newly obtained states.
            self._population_rewards[self._survivors_count :] += reward_batch
            weights = self._discount_factor ** np.arange(episode_action_count, -1, -1)
            self._population_returns[
                : episode_action_count + 1, self._survivors_count :
            ] += np.outer(weights, reward_batch)
            episode_action_count += 1
            self._population_states[episode_action_count, self._survivors_count :, :] = state_batch

        # Initialize the mask that decides which executed episodes should be used to train the
        # action prediction model.
        elite_mask = np.zeros((self._survivors_count + self._new_candidates_count), dtype=bool)
        # Initialize the mask that decides which executed episodes should be carried over to the
        # next generation.
        survivors_mask = np.zeros((self._survivors_count + self._new_candidates_count), dtype=bool)

        # In the first iteration of the learning process, only the last `_new_candidates_count`
        # executed episodes should be considered because there are no survivor episodes. Therefore,
        # the first `_survivors_count` episodes correspond to nonexisting episodes, hence they
        # should be ignored.
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
        # In any subsequent iteration of the learning process, there are survivor episodes from the
        # previous iteration, so all the episodes from the `np.ndarray` are valid, which means that
        # they should all be considered.
        else:
            elite_mask[
                np.argpartition(self._population_rewards, -self._elite_count)[-self._elite_count :]
            ] = True
            survivors_mask[
                np.argpartition(self._population_rewards, -self._survivors_count)[
                    -self._survivors_count :
                ]
            ] = True

        # Extract the (state, action) pairs from the elite executed episodes, i.e., the executed
        # episodes that should be used to train the action prediction model.
        elite_states = self._population_states[:-1, elite_mask, :].reshape(
            -1, self._environment.state_length
        )
        elite_actions = self._population_actions[:, elite_mask].reshape(-1)
        elite_returns = self._population_returns[:, elite_mask].reshape(-1)

        elite_states_torch = torch.from_numpy(elite_states.astype(np.float32)).to(self._device)
        elite_actions_torch = torch.from_numpy(elite_actions.astype(np.int64)).to(self._device)
        elite_returns_torch = torch.from_numpy(elite_returns.astype(np.int64)).to(self._device)

        baseline = np.mean(self._population_returns[0, elite_mask]).item()

        # Recalculate log probs with gradient tracking
        self._optimizer.zero_grad()
        logits_torch = self._policy_network(elite_states_torch)
        log_probs_torch = Categorical(logits=logits_torch).log_prob(elite_actions_torch)

        # REINFORCE: policy gradient = -log_prob * (return - baseline)
        # Negative because we want to maximize return, but optimizer minimizes loss
        advantages_torch = elite_returns_torch - baseline
        loss = -(log_probs_torch * advantages_torch).mean()

        # Optimization step

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy_network.parameters(), 0.5)
        self._optimizer.step()

        # Extract the states, actions and rewards from the survivor episodes and store them so that
        # they are available in the next generation.
        self._population_states[:, : self._survivors_count, :] = self._population_states[
            :, survivors_mask, :
        ]
        self._population_actions[:, : self._survivors_count] = self._population_actions[
            :, survivors_mask
        ]
        self._population_rewards[: self._survivors_count] = self._population_rewards[
            survivors_mask
        ]
        self._population_returns[:, : self._survivors_count] = self._population_returns[
            survivors_mask
        ]

        # Update the best score, and the random action probability through the random action
        # mechanism.
        new_best_score = np.max(self._population_rewards[: self._survivors_count]).item()
        self._random_action_mechanism.step(
            previous_best_score=self._best_score, new_best_score=new_best_score
        )
        self._best_score = new_best_score

        # Increment the number of executed iterations of the learning process.
        self._step_count += 1

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def best_graph(self) -> Optional[Graph]:
        if self._step_count is None or self._step_count < 1:
            return None

        # Run one episode with the current policy to get the best graph
        state_batch, status = self._environment.reset_batch(batch_size=1)

        while status == EpisodeStatus.IN_PROGRESS:
            state_torch = torch.from_numpy(state_batch.astype(np.float32)).to(self._device)

            with torch.no_grad():
                logits = self._policy_network(state_torch)

                action_mask = self._environment.action_mask
                if action_mask is not None:
                    action_mask_torch = torch.from_numpy(action_mask).to(self._device)
                    logits = logits.masked_fill(~action_mask_torch, float("-inf"))

                action = Categorical(logits=logits).sample()

            state_batch, reward_batch, status = self._environment.step_batch(action.cpu().numpy())

        # Convert final state to graph
        final_state = state_batch[0]
        best_graph = self._environment.state_to_graph(final_state)

        return best_graph
