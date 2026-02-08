"""
This ``Python`` module contains the `PPOAgent` class, which encapsulates the concept
of a reinforcement learning agent to be used in graph theory applications that applies the
``PyTorch``-based Proximal Policy Optimization (PPO) method.
"""

from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..environments.graph_environment import EpisodeStatus, GraphEnvironment
from ..graphs.graph import Graph
from .graph_agent import GraphAgent
from .random_action_mechanisms import NoRandomActionMechanism, RandomActionMechanism


class PPOAgent(GraphAgent):
    """
    This class encapsulates the concept of an RL agent to be used in graph theory applications that
    applies the ``PyTorch``-based Proximal Policy Optimization (PPO) method. The agent operates over
    a configurable environment given as a `GraphEnvironment` object. In each iteration of the learning
    process, the agent generates a predetermined number of graphs through the graph-building game induced
    by the environment and computes the cumulative reward for each of these episodes. The game is played
    by using actor-critic architecture with a policy network (`torch.nn.Module`) and a value network
    (`torch.nn.Module`). The policy network computes action probabilities, while the value network
    estimates state values. The agent uses the PPO algorithm to update both networks based on collected
    experiences. Additionally, the user can configure the optimizer that trains both networks, given as a
    `torch.optim.Optimizer` object. Finally, the user can also configure the random action mechanism
    that the RL agent should use. When a random action is supposed to get executed, it gets sampled
    with the uniform probability distribution over all the actions that are available for execution.

    :ivar _environment: A `GraphEnvironment` object that represents the RL environment that defines
        the extremal problem of interest and whose graph-building game should be used to construct
        all the graphs.
    :ivar _policy_network: A `torch.nn.Module` object that represents the policy network (actor)
        that is used to compute the probability for each of the actions to be selected.
    :ivar _value_network: A `torch.nn.Module` object that represents the value network (critic)
        that estimates the state values for advantage calculation.
    :ivar _optimizer: A `torch.optim.Optimizer` object that represents the optimizer responsible
        for updating both network parameters.
    :ivar _device: A `torch.device` object that determines on which device the networks are located.
    :ivar _batch_size: A positive `int` that determines how many episodes should be collected before
        performing a policy update.
    :ivar _gamma: A `float` in the range [0, 1] representing the discount factor for returns.
    :ivar _eps_clip: A positive `float` representing the PPO clipping parameter.
    :ivar _k_epochs: A positive `int` determining how many epochs to perform during each PPO update.
    :ivar _entropy_coef: A nonnegative `float` representing the entropy coefficient for exploration.
    :ivar _value_coef: A nonnegative `float` representing the value loss coefficient.
    :ivar _random_action_mechanism: A `RandomActionMechanism` object that indicates the random
        action mechanism that the RL agent should use.
    :ivar _rng: The `numpy.random.Generator` object that represents the random number generator
        used for all the probabilistic decisions.
    :ivar _step_count: A nonnegative `int` that determines the number of executed iterations of the
        learning process, if the RL agent has been initialized, and otherwise, `None`.
    :ivar _best_score: A `float` that determines the best currently achieved value for the graph
        invariant that is supposed to get maximized in the configured extremal problem, if the RL
        agent has been initialized, and otherwise, `None`.
    :ivar _buffer_states: A list used to store states during episode collection.
    :ivar _buffer_actions: A list used to store actions during episode collection.
    :ivar _buffer_rewards: A list used to store rewards during episode collection.
    :ivar _buffer_log_probs: A list used to store log probabilities during episode collection.
    :ivar _buffer_values: A list used to store value estimates during episode collection.
    :ivar _buffer_dones: A list used to store episode termination flags during episode collection.
    """

    def __init__(
        self,
        environment: GraphEnvironment,
        policy_network: nn.Module,
        value_network: nn.Module,
        optimizer: torch.optim.Optimizer,
        candidates_count: int = 200,
        elite_count: Optional[int] = None,
        discount_factor: float = 0.99,
        epochs_count: int = 4,
        clamp_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        random_action_mechanism: RandomActionMechanism = NoRandomActionMechanism(),
        random_generator: Optional[np.random.Generator] = None,
    ):
        """
        This constructor initializes an instance of the `PPOAgent` class.

        :param environment: The RL environment that defines the extremal problem of interest and
            whose graph-building game should be used to construct all the graphs, given as a
            `GraphEnvironment` object.
        :param policy_network: The policy network (actor) that is used to compute the probability
            for each of the actions to be selected, given as a `torch.nn.Module` object.
        :param value_network: The value network (critic) that estimates state values for advantage
            calculation, given as a `torch.nn.Module` object.
        :param optimizer: The optimizer responsible for updating both network parameters, given as a
            `torch.optim.Optimizer` object. The parameters of both the policy network and value
            network must be passed to the optimizer.
        :param batch_size: A positive `int` that determines how many episodes should be collected
            before performing a policy update. The default value is 32.
        :param gamma: A `float` in the range [0, 1] representing the discount factor for returns.
            The default value is 0.99.
        :param eps_clip: A positive `float` representing the PPO clipping parameter. The default
            value is 0.2.
        :param k_epochs: A positive `int` determining how many epochs to perform during each PPO
            update. The default value is 4.
        :param entropy_coef: A nonnegative `float` representing the entropy coefficient for
            exploration. The default value is 0.01.
        :param value_coef: A nonnegative `float` representing the value loss coefficient. The
            default value is 0.5.
        :param random_action_mechanism: The random action mechanism that the RL agent should use,
            given as a `RandomActionMechanism` object. The default value is
            ``NoRandomActionMechanism()``, i.e., no random actions should be executed by default.
        :param rng: Either `None`, or the `numpy.random.Generator` object that represents the
            random number generator used for all the probabilistic decisions. If this argument is
            `None`, then a default `numpy.random.Generator` object will be used. The default value
            is `None`.
        """

        self._environment: GraphEnvironment = environment
        # Enforce no sparse setting.
        self._environment.sparse_setting = False

        self._policy_network: nn.Module = policy_network
        self._value_network: nn.Module = value_network
        self._optimizer: torch.optim.Optimizer = optimizer
        self._value_loss_function: Callable = nn.MSELoss()

        # Infer the device from the policy network parameters. If no parameters exist, fall back to
        # the CPU.
        params = list(self._policy_network.parameters())
        self._device: torch.device = params[0].device if params else torch.device("cpu")

        self._candidates_count: int = candidates_count
        self._elite_count: Optional[int] = elite_count
        self._discount_factor: float = discount_factor

        self._epochs_count: int = epochs_count
        self._clamp_epsilon: float = clamp_epsilon
        self._value_loss_coef: float = value_loss_coef
        self._random_action_mechanism: RandomActionMechanism = random_action_mechanism

        # If the ``random_generator`` argument is `None`, then use a default `np.random.Generator`.
        if random_generator is None:
            random_generator = np.random.default_rng()
        self._random_generator: np.random.Generator = random_generator

        self._step_count: Optional[int] = None
        self._best_score: Optional[float] = None
        self._best_graph: Optional[Graph] = None

        self._population_states: Optional[np.ndarray] = None
        self._population_actions: Optional[np.ndarray] = None
        self._population_returns: Optional[np.ndarray] = None

    def reset(self) -> None:
        # Initialize the step count to 0, the best score to minus infinity, and the best graph to
        # `None`. Also, initialize the random action mechanism.
        self._step_count = 0
        self._best_score = float("-inf")
        self._best_graph = None
        self._random_action_mechanism.reset()

        # Initialize the population returns to the zero `np.ndarray` of type `np.float32` and the
        # required shape.
        self._population_states = np.zeros(
            (
                self._environment.episode_length + 1,
                self._candidates_count,
                self._environment.state_length,
            ),
            dtype=self._environment.state_dtype,
        )
        self._population_actions = np.zeros(
            (self._environment.episode_length, self._candidates_count), dtype=np.int32
        )
        self._population_returns = np.zeros(
            (self._environment.episode_length, self._candidates_count), dtype=np.float32
        )

    def step(self) -> None:
        # Initialize a batch of episodes with the batch size ``_candidates_count``.
        state_batch, current_scores, status = self._environment.reset_batch(
            batch_size=self._candidates_count
        )
        self._population_states[0, :, :] = state_batch

        # Initialize the new best score and new best graph.
        new_best_score = self._best_score
        new_best_graph = self._best_graph

        # If the RL environment is continuing, then determine the best graph and best score from
        # the starting timestamp, and update the corresponding variables if needed.
        if self._environment.is_continuing:
            timestamp_best = np.max(current_scores)
            if timestamp_best > new_best_score:
                new_best_score = timestamp_best
                best_index = np.argmax(current_scores)
                new_best_graph = self._environment.state_to_graph(state_batch[best_index, :])

        # Set the ``_population_returns`` attribute to all zeros. Also, initialize the
        # ``population_log_probs`` list that stores all the log probabilities per timestamp.
        self._population_returns[:, :] = 0
        population_old_log_probs = []
        population_old_values = []

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
            
            values_batch_torch = self._value_network(state_batch_torch)
            population_old_values.append(values_batch_torch.detach())

            # Make it impossible to execute an action that is not available for execution.
            action_mask = self._environment.action_mask
            if action_mask is not None:
                action_mask_torch = torch.from_numpy(action_mask).to(self._device)
                logits_batch_torch = logits_batch_torch.masked_fill(
                    ~action_mask_torch, float("-inf")
                )

            # Sample the actions according to the obtained probability distributions.
            distribution = Categorical(logits=logits_batch_torch)
            action_batch_torch = distribution.sample()
            action_batch = action_batch_torch.cpu().numpy()

            # Store the log probabilities to the ``population_log_probs`` list.
            population_old_log_probs.append(distribution.log_prob(action_batch_torch).detach())

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

            # Execute the selected actions and compute the batch of rewards.
            previous_scores = current_scores
            self._population_actions[episode_action_count, :] = action_batch
            state_batch, current_scores, status = self._environment.step_batch(action_batch)
            reward_batch = current_scores - previous_scores

            # Update the discounted returns.
            weights = self._discount_factor ** np.arange(episode_action_count, -1, -1)
            self._population_returns[: episode_action_count + 1, :] += np.outer(
                weights, reward_batch
            )

            # If the RL environment is continuing, or the final batch of actions has been executed,
            # then determine the best graph and best score from the current timestamp, and update
            # the corresponding variables if needed.
            if self._environment.is_continuing or status != EpisodeStatus.IN_PROGRESS:
                timestamp_best = np.max(current_scores)
                if timestamp_best > new_best_score:
                    new_best_score = timestamp_best
                    best_index = np.argmax(current_scores)
                    new_best_graph = self._environment.state_to_graph(state_batch[best_index, :])

            episode_action_count += 1
            self._population_states[episode_action_count, :, :] = state_batch

        # Compute the mask that decides which executed episodes should be used to train the policy
        # network and the value network.
        elite_mask = np.zeros((self._candidates_count), dtype=bool)
        if self._elite_count is None:
            elite_mask[:] = True
        else:
            elite_mask[
                np.argpartition(current_scores, -self._elite_count)[-self._elite_count :]
            ] = True

        # Extract the (state, action) pairs from the elite executed episodes, i.e., the executed
        # episodes that should be used to train the policy network.
        elite_states = self._population_states[:-1, elite_mask, :].reshape(
            -1, self._environment.state_length
        )
        elite_actions = self._population_actions[:, elite_mask].reshape(-1)
        elite_states_torch = torch.from_numpy(elite_states.astype(np.float32)).to(self._device)
        elite_actions_torch = torch.from_numpy(elite_actions.astype(np.int64)).to(self._device)

        # Prepare the log-probabilities from the elite executed episodes for training.
        elite_log_probs_torch = torch.cat(
            [
                population_old_log_probs[index][elite_mask]
                for index in range(self._environment.episode_length)
            ]
        ).reshape(-1)

        elite_values_torch = torch.cat(
            [
                population_old_values[index][elite_mask]
                for index in range(self._environment.episode_length)
            ]
        ).reshape(-1)

        # Compute the advantages for the elite executed episodes. Apply the baselines if requested.
        elite_returns = self._population_returns[:, elite_mask].reshape(-1)
        elite_returns_torch = torch.from_numpy(elite_returns.astype(np.float32)).to(self._device)

        elite_advantages_torch = elite_returns_torch - elite_values_torch
        elite_advantages_torch = (elite_advantages_torch - elite_advantages_torch.mean()) / (elite_advantages_torch.std() + 1e-8)

        for _ in range(self._epochs_count):
            logits_batch_torch = self._policy_network(elite_states_torch)
            values_batch_torch = self._value_network(elite_states_torch).reshape(-1)

            new_log_probs_torch = Categorical(logits=logits_batch_torch).log_prob(elite_actions_torch)
            ratio = torch.exp(new_log_probs_torch - elite_log_probs_torch)
            loss_1 = ratio * elite_advantages_torch
            loss_2 = torch.clamp(ratio, 1 - self._clamp_epsilon, 1 + self._clamp_epsilon) * elite_advantages_torch
            policy_loss = torch.min(loss_1, loss_2).mean()

            value_loss = self._value_loss_function(values_batch_torch, elite_returns_torch)
            loss = policy_loss + self._value_loss_coef * value_loss

            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self._policy_network.parameters()) + list(self._value_network.parameters()), 0.5)
            self._optimizer.step()

        # Update the random action probability through the random action mechanism, and then update
        # the best score and best graph.
        self._random_action_mechanism.step(
            previous_best_score=self._best_score, current_best_score=new_best_score
        )
        self._best_score = new_best_score
        self._best_graph = new_best_graph

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
        return self._best_graph