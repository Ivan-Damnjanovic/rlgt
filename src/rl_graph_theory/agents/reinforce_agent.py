"""
This ``Python`` module contains the `ReinforceAgent` class, which encapsulates the concept
of a reinforcement learning agent to be used in graph theory applications that applies the
``PyTorch``-based REINFORCE algorithm (Monte Carlo Policy Gradient).
"""

from typing import Optional

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
        policy_network: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_size: int = 32,
        gamma: float = 0.99,
        use_baseline: bool = True,
        elite_fraction: float = 1.0,
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
        
        self._policy_network: nn.Module = policy_network
        self._optimizer: torch.optim.Optimizer = optimizer

        # Infer the device from the policy network parameters. If no parameters exist, fall back to
        # the CPU.
        params = list(self._policy_network.parameters())
        self._device: torch.device = params[0].device if params else torch.device("cpu")
        
        self._batch_size: int = batch_size
        self._gamma: float = gamma
        self._use_baseline: bool = use_baseline
        self._elite_fraction: float = elite_fraction

        self._random_action_mechanism: RandomActionMechanism = random_action_mechanism

        # If the ``rng`` argument is `None`, then use a default `np.random.Generator`.
        if rng is None:
            rng = np.random.default_rng()
        self._rng: np.random.Generator = rng
        
        self._step_count: Optional[int] = None
        self._best_score: Optional[float] = None

    def reset(self) -> None:
        """Reset the agent for a new training session."""
        self._step_count = 0
        self._best_score = float("-inf")
        self._random_action_mechanism.reset()
    
    def step(self) -> None:
        """
        Perform one training step: collect batch_size episodes in parallel and update policy using REINFORCE.
        """
        # Get the random action probability from the random action mechanism
        random_action_probability = self._random_action_mechanism.random_action_probability
        
        # Collect batch_size episodes IN PARALLEL
        state_batch, status = self._environment.reset_batch(batch_size=self._batch_size)
        episode_rewards = np.zeros(self._batch_size, dtype=np.float32)
        
        # Store per-episode trajectories for REINFORCE
        episode_trajectories = [[] for _ in range(self._batch_size)]
        
        while status == EpisodeStatus.IN_PROGRESS:
            # Select actions for all active episodes
            state_batch_torch = torch.from_numpy(state_batch.astype(np.float32)).to(self._device)
            
            with torch.no_grad():
                logits_batch = self._policy_network(state_batch_torch)
                
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
                        [self._rng.choice(probabilities_batch.shape[1], p=probs) 
                         for probs in probabilities_batch],
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
            # Store states and actions for later log_prob recalculation
            for i in range(self._batch_size):
                episode_trajectories[i].append({
                    'state': state_batch_torch[i].detach(),
                    'action': action_batch[i],
                })
            
            # Take actions in environment
            next_state_batch, reward_batch, status = self._environment.step_batch(action_batch)
            
            # Update episode rewards and store rewards per episode
            for i in range(self._batch_size):
                episode_trajectories[i][-1]['reward'] = reward_batch[i]
                episode_rewards[i] += reward_batch[i]
            
            state_batch = next_state_batch
        
        # Update policy using REINFORCE algorithm
        self._update_policy(episode_trajectories, episode_rewards)
        
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
    
    def _update_policy(self, episode_trajectories, episode_rewards):
        """Update policy network using REINFORCE algorithm."""
        if len(episode_trajectories) == 0:
            return
        
        # Elite filtering: only use top fraction of episodes if elite_fraction < 1.0
        if self._elite_fraction < 1.0:
            elite_count = max(1, int(self._batch_size * self._elite_fraction))
            elite_indices = np.argpartition(episode_rewards, -elite_count)[-elite_count:]
            episode_trajectories = [episode_trajectories[i] for i in elite_indices]
            episode_rewards = episode_rewards[elite_indices]
        
        # Calculate baseline (average return) if using baseline
        baseline = np.mean(episode_rewards) if self._use_baseline else 0.0
        
        # Collect all states, actions, and returns from all episodes
        all_states = []
        all_actions = []
        all_returns = []
        
        for ep_idx, trajectory in enumerate(episode_trajectories):
            # Calculate returns for this episode
            returns = []
            discounted_sum = 0
            for transition in reversed(trajectory):
                discounted_sum = transition['reward'] + self._gamma * discounted_sum
                returns.insert(0, discounted_sum)
            
            # Store states, actions, and returns
            for t, transition in enumerate(trajectory):
                all_states.append(transition['state'])
                all_actions.append(transition['action'])
                all_returns.append(returns[t])
        
        # Convert to tensors
        states = torch.stack(all_states).to(self._device)
        actions = torch.tensor(all_actions, dtype=torch.int64).to(self._device)
        returns = torch.tensor(all_returns, dtype=torch.float32).to(self._device)
        
        # Recalculate log probs with gradient tracking
        logits = self._policy_network(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        
        # REINFORCE: policy gradient = -log_prob * (return - baseline)
        # Negative because we want to maximize return, but optimizer minimizes loss
        advantages = returns - baseline
        policy_loss = -(log_probs * advantages).mean()
        
        # Optimization step
        self._optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy_network.parameters(), 0.5)
        self._optimizer.step()
    
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
