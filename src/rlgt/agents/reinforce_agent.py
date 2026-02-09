"""
This ``Python`` module contains the `ReinforceAgent` class, which implements a reinforcement
learning agent for graph theory applications using the REINFORCE method with ``PyTorch``.
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
    This class encapsulates a reinforcement learning agent for graph theory applications using the
    ``PyTorch``-based REINFORCE method. The agent operates on a configurable environment given as a
    `GraphEnvironment` object. In each iteration of the learning process, the agent generates a
    predetermined number of graphs by playing the graph building game defined by the environment
    and computes the graph invariant values and all discounted returns for each episode run in
    parallel. Here, while computing a discounted return, a reward is considered to be the increase
    between two consecutive graph invariant values. The agent uses a `torch.nn.Module` model to
    compute the probability of selecting each action in each step of every episode. Afterwards, the
    log probabilities and discounted returns of a subset of top-performing episodes are used to
    train the model according to the REINFORCE algorithm. This completes one iteration of the
    learning process. The user provides the model, configures the optimizer, sets the discount
    factor, decides whether to apply a baseline to reduce variance, and optionally provides a
    random action mechanism. When a random action occurs, it is selected uniformly among all
    actions available in the current state.

    :ivar _environment: A `GraphEnvironment` object defining the extremal problem and providing the
        graph building game used to construct all the graphs.
    :ivar _policy_network: A `torch.nn.Module` object predicting the action probabilities for each
        step in each episode.
    :ivar _optimizer: A `torch.optim.Optimizer` object that updates the model parameters.
    :ivar _device: A `torch.device` object indicating the device where the model resides.
    :ivar _candidates_count: A positive `int` specifying the number of graphs constructed per
        iteration, i.e., the number of episodes run in parallel.
    :ivar _elite_count: A positive `int` specifying the number of top-performing episodes used to
        train the model in each iteration, or `None` if all the episodes should be used.
    :ivar _discount_factor: A `float` from the interval [0, 1] representing the discount factor to
        be used while computing the discounted returns.
    :ivar _apply_baseline: A `bool` indicating whether a baseline should be applied to reduce
        variance. If `True`, the baseline is the mean return over all episodes, computed
        independently for each step.
    :ivar _random_action_mechanism: A `RandomActionMechanism` object that determines the
        probability of executing a random action. When a random action is selected, it is sampled
        uniformly among all available actions in the current state.
    :ivar _random_generator: A `numpy.random.Generator` object used for all probabilistic
        decisions.
    :ivar _step_count: A nonnegative `int` representing the number of executed iterations, or
        `None` if the agent has not been initialized.
    :ivar _best_score: A `float` representing the best achieved value for the graph invariant, or
        `None` if the agent has not been initialized.
    :ivar _best_graph: A `Graph` object representing a graph attaining the best achieved value for
        the graph invariant, or `None` if the agent has not been initialized or no iterations have
        been executed.
    :ivar _population_returns: Either `None` if uninitialized, or a `numpy.ndarray` of type
        `numpy.float32` storing the discounted returns for all executed episodes. Its shape is
        ``(episode_length, candidates_count)``, where ``episode_length`` is the episode length of
        the RL environment and ``candidates_count`` is the number of episodes executed in parallel.
        The first dimension corresponds to the timestamps (actions) within an episode, and the
        second corresponds to the executed episodes.
    """

    def __init__(
        self,
        environment: GraphEnvironment,
        policy_network: nn.Module,
        optimizer: torch.optim.Optimizer,
        candidates_count: int = 200,
        elite_count: Optional[int] = None,
        discount_factor: float = 0.99,
        apply_baseline: bool = True,
        random_action_mechanism: RandomActionMechanism = NoRandomActionMechanism(),
        random_generator: Optional[np.random.Generator] = None,
    ):
        """
        This constructor initializes an instance of the `ReinforceAgent` class.

        :param environment: The RL environment defining the extremal problem and providing the
            graph building game, given as a `GraphEnvironment` object.
        :param policy_network: The policy network used to compute the probability of each action in
            each episode and step, given as a `torch.nn.Module` object.
        :param optimizer: The optimizer responsible for updating the model parameters, given as a
            `torch.optim.Optimizer` object. The parameters of ``policy_network`` must be passed to
            it.
        :param candidates_count: A positive `int` specifying how many graphs are generated in each
            iteration by running the corresponding number of episodes in parallel. The default
            value is 200.
        :param elite_count: A positive `int` specifying how many episodes with the greatest graph
            invariant value are used to train the policy network in each iteration of the learning
            process, or `None` to indicate that all executed episodes should be used. The default
            value is `None`.
        :param discount_factor: A `float` from the interval [0, 1] representing the discount factor
            to be used while computing the returns. The default value is 0.99.
        :param apply_baseline: A `bool` indicating whether a baseline should be applied to reduce
            variance. If `True`, the baseline is the mean return over all elite episodes, computed
            independently for each step. The default value is `True`.
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
        # Enforce no sparse setting.
        self._environment.sparse_setting = False

        self._policy_network: nn.Module = policy_network
        self._optimizer: torch.optim.Optimizer = optimizer

        # Infer the device from the policy network parameters. If no parameters exist, fall back to
        # the CPU.
        params = list(self._policy_network.parameters())
        self._device: torch.device = params[0].device if params else torch.device("cpu")

        self._candidates_count: int = candidates_count
        self._elite_count: Optional[int] = elite_count
        self._discount_factor: float = discount_factor
        self._apply_baseline: bool = apply_baseline
        self._random_action_mechanism: RandomActionMechanism = random_action_mechanism

        # If the ``random_generator`` argument is `None`, then use a default `np.random.Generator`.
        if random_generator is None:
            random_generator = np.random.default_rng()
        self._random_generator: np.random.Generator = random_generator

        self._step_count: Optional[int] = None
        self._best_score: Optional[float] = None
        self._best_graph: Optional[Graph] = None
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
        self._population_returns = np.zeros(
            (self._environment.episode_length, self._candidates_count), dtype=np.float32
        )

    def step(self) -> None:
        # Initialize a batch of episodes with the batch size ``_candidates_count``.
        state_batch, current_scores, status = self._environment.reset_batch(
            batch_size=self._candidates_count
        )

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
        population_log_probs = []

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
            distribution = Categorical(logits=logits_batch_torch)
            action_batch_torch = distribution.sample()
            action_batch = action_batch_torch.cpu().numpy()

            # Store the log probabilities to the ``population_log_probs`` list.
            population_log_probs.append(distribution.log_prob(action_batch_torch))

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

        # Compute the mask that decides which executed episodes should be used to train the policy
        # network.
        elite_mask = np.zeros((self._candidates_count), dtype=bool)
        if self._elite_count is None:
            elite_mask[:] = True
        else:
            elite_mask[
                np.argpartition(current_scores, -self._elite_count)[-self._elite_count :]
            ] = True

        # Compute the advantages for the elite executed episodes. Apply the baselines if requested.
        elite_advantages = self._population_returns[:, elite_mask]
        if self._apply_baseline:
            elite_advantages = elite_advantages - elite_advantages.mean(axis=1, keepdims=True)
        elite_advantages = elite_advantages.reshape(-1)
        elite_advantages_torch = torch.from_numpy(elite_advantages.astype(np.float32)).to(
            self._device
        )

        # Prepare the log probabilities from the elite executed episodes for training.
        elite_log_probs_torch = torch.cat(
            [
                population_log_probs[index][elite_mask]
                for index in range(self._environment.episode_length)
            ]
        ).reshape(-1)

        # Use the optimizer to train the policy network.
        self._optimizer.zero_grad()
        loss = -(elite_log_probs_torch * elite_advantages_torch).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy_network.parameters(), 0.5)
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
