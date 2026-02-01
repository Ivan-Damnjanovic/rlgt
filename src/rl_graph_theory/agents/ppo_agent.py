"""
This ``Python`` module contains the `PPOAgent` class, which encapsulates the concept
of a reinforcement learning agent to be used in graph theory applications that applies the
``PyTorch``-based Proximal Policy Optimization (PPO) method.
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
        batch_size: int = 32,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        random_action_mechanism: RandomActionMechanism = NoRandomActionMechanism(),
        rng: Optional[np.random.Generator] = None,
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

        self._policy_network: nn.Module = policy_network
        self._value_network: nn.Module = value_network
        self._optimizer: torch.optim.Optimizer = optimizer

        # Infer the device from the policy network parameters. If no parameters exist, fall back to
        # the CPU.
        params = list(self._policy_network.parameters())
        self._device: torch.device = params[0].device if params else torch.device("cpu")

        self._batch_size: int = batch_size
        self._gamma: float = gamma
        self._eps_clip: float = eps_clip
        self._k_epochs: int = k_epochs
        self._entropy_coef: float = entropy_coef
        self._value_coef: float = value_coef

        self._random_action_mechanism: RandomActionMechanism = random_action_mechanism

        # If the ``rng`` argument is `None`, then use a default `np.random.Generator`.
        if rng is None:
            rng = np.random.default_rng()
        self._rng: np.random.Generator = rng

        self._step_count: Optional[int] = None
        self._best_score: Optional[float] = None

        # Episode buffer for collecting experiences
        self._buffer_states = []
        self._buffer_actions = []
        self._buffer_rewards = []
        self._buffer_log_probs = []
        self._buffer_values = []
        self._buffer_dones = []

    def reset(self) -> None:
        """Reset the agent for a new training session."""
        # Initialize the step count to 0 and the best score to minus infinity. Also, initialize the
        # random action mechanism.
        self._step_count = 0
        self._best_score = float("-inf")
        self._random_action_mechanism.reset()
        self._clear_buffer()

    def step(self) -> None:
        """
        Perform one training step: collect batch_size episodes in parallel and update policy.
        Episodes are collected in parallel for speed, but episode boundaries are tracked
        to ensure correct per-episode returns calculation.
        """
        # Get the random action probability from the random action mechanism
        random_action_probability = self._random_action_mechanism.random_action_probability

        # Collect batch_size episodes IN PARALLEL
        # Track episode boundaries for correct returns calculation
        state_batch, status = self._environment.reset_batch(batch_size=self._batch_size)
        episode_rewards = np.zeros(self._batch_size, dtype=np.float32)
        episode_active = np.ones(
            self._batch_size, dtype=bool
        )  # Track which episodes are still running

        # Store per-episode trajectories to calculate returns correctly
        episode_trajectories = [[] for _ in range(self._batch_size)]

        while status == EpisodeStatus.IN_PROGRESS:
            # Select actions for all active episodes
            state_batch_torch = torch.from_numpy(state_batch.astype(np.float32)).to(self._device)

            with torch.no_grad():
                logits_batch = self._policy_network(state_batch_torch)
                values_batch = self._value_network(state_batch_torch)

                # Apply action mask if available
                action_mask = self._environment.action_mask
                if action_mask is not None:
                    action_mask_torch = torch.from_numpy(action_mask).to(self._device)
                    logits_batch = logits_batch.masked_fill(~action_mask_torch, float("-inf"))

                dist = Categorical(logits=logits_batch)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            action_batch = actions.cpu().numpy()

            # Apply random actions with configured probability
            random_mask = self._rng.random(size=self._batch_size) < random_action_probability
            if np.any(random_mask):
                if action_mask is not None:
                    probabilities_batch = action_mask[random_mask].astype(np.float32)
                    probabilities_batch /= probabilities_batch.sum(axis=1, keepdims=True)
                    action_batch[random_mask] = np.array(
                        [
                            self._rng.choice(probabilities_batch.shape[1], p=probs)
                            for probs in probabilities_batch
                        ],
                        dtype=np.int32,
                    )
                else:
                    action_batch[random_mask] = self._rng.integers(
                        low=0,
                        high=self._environment.action_number,
                        size=np.count_nonzero(random_mask),
                        dtype=np.int32,
                    )

            # Store transitions for each episode separately
            for i in range(self._batch_size):
                episode_trajectories[i].append(
                    {
                        "state": state_batch_torch[i],
                        "action": action_batch[i],
                        "log_prob": log_probs[i].item(),
                        "value": values_batch[i].item(),
                    }
                )

            # Take actions in environment
            next_state_batch, reward_batch, status = self._environment.step_batch(action_batch)

            # Update episode rewards and store rewards/dones per episode
            for i in range(self._batch_size):
                episode_trajectories[i][-1]["reward"] = reward_batch[i]
                episode_trajectories[i][-1]["done"] = status != EpisodeStatus.IN_PROGRESS
                episode_rewards[i] += reward_batch[i]

            state_batch = next_state_batch

        # ELITE FILTERING: Only use top episodes for training (like Cross Entropy Method)
        # This dramatically improves learning for sparse reward problems
        elite_count = max(20, self._batch_size // 10)  # Top 10% or at least 20
        elite_indices = np.argpartition(episode_rewards, -elite_count)[-elite_count:]

        # Only flatten elite episode trajectories into the buffer
        for ep_idx in elite_indices:
            trajectory = episode_trajectories[ep_idx]
            for transition in trajectory:
                self._buffer_states.append(transition["state"])
                self._buffer_actions.append(transition["action"])
                self._buffer_log_probs.append(transition["log_prob"])
                self._buffer_values.append(transition["value"])
                self._buffer_rewards.append(transition["reward"])
                self._buffer_dones.append(transition["done"])

        # Update policy using collected experiences from elite episodes only
        self._update_policy()

        # Track best score and update random action mechanism
        max_episode_score = float(max(episode_rewards))
        previous_best_score = self._best_score
        if max_episode_score > self._best_score:
            self._best_score = max_episode_score

        # Call random action mechanism step - try both parameter names for compatibility
        try:
            self._random_action_mechanism.step(
                previous_best_score=previous_best_score, current_best_score=self._best_score
            )
        except TypeError:
            # Fallback for ExponentialRandomActionMechanism which uses new_best_score
            self._random_action_mechanism.step(
                previous_best_score=previous_best_score, new_best_score=self._best_score
            )

        # Increment the number of executed iterations of the learning process.
        self._step_count += 1

    def _update_policy(self):
        """Update policy and value networks using PPO algorithm."""
        if len(self._buffer_states) == 0:
            return

        # Convert buffer to tensors
        states = torch.stack(self._buffer_states).to(self._device)
        actions = torch.tensor(self._buffer_actions, dtype=torch.int64).to(self._device)
        old_log_probs = torch.tensor(self._buffer_log_probs, dtype=torch.float32).to(self._device)

        # Calculate returns
        returns = self._calculate_returns()
        returns = torch.tensor(returns, dtype=torch.float32).to(self._device)

        # DON'T normalize returns for sparse rewards - it destroys the signal!
        # Elite filtering already selects good episodes, so normalization is harmful

        # PPO update for k epochs
        for _ in range(self._k_epochs):
            # Get current policy and value predictions
            logits = self._policy_network(states)
            values = self._value_network(states).squeeze()

            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Calculate advantages WITHOUT normalization for sparse rewards
            # Normalizing advantages destroys the signal when rewards are sparse
            advantages = returns - values.detach()

            # Calculate surrogate losses
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self._eps_clip, 1 + self._eps_clip) * advantages

            # Total loss
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values, returns)
            loss = policy_loss + self._value_coef * value_loss - self._entropy_coef * entropy

            # Optimization step
            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self._policy_network.parameters()) + list(self._value_network.parameters()),
                0.5,
            )
            self._optimizer.step()

        # Clear buffer
        self._clear_buffer()

    def _calculate_returns(self):
        """Calculate discounted returns from rewards."""
        returns = []
        discounted_sum = 0

        for reward, done in zip(reversed(self._buffer_rewards), reversed(self._buffer_dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self._gamma * discounted_sum
            returns.insert(0, discounted_sum)

        return returns

    def _clear_buffer(self):
        """Clear experience buffer."""
        self._buffer_states = []
        self._buffer_actions = []
        self._buffer_rewards = []
        self._buffer_log_probs = []
        self._buffer_values = []
        self._buffer_dones = []

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
