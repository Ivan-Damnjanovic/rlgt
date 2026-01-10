"""
This ``Python`` module contains the `DeepCrossEntropyMethod` class, which encapsulates the concept
of a reinforcement learning agent to be used in graph theory applications that applies the
``PyTorch``-based deep cross entropy method.
"""

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..environments.graph_environment import EpisodeStatus, GraphEnvironment
from ..environments.graph_generators import create_fixed_graph_generator
from ..graphs.graph import Graph
from ..graphs.graph_formats import GraphFormat, FlattenedOrdering
from .random_action_mechanisms import RandomActionMechanism


class DeepCrossEntropyMethod:
    """
    This class encapsulates the concept of an RL agent to be used in graph theory applications that
    applies the ``PyTorch``-based deep cross entropy method. The agent operates over a configurable
    environment given as a `GraphEnvironment` object. In each iteration, the agent generates a
    predetermined number of graphs through the graph-building game induced by the environment and
    computes the cumulative reward for each of these episodes run in parallel. The game is played
    by using a `torch.nn.Module` model to compute the probability for each of the actions to be
    selected in each episode and in each step. Afterwards, a certain number of episodes with the
    greatest cumulative reward are used to train the model, while another predetermined number of
    episodes with the greatest cumulative reward is carried over to the next generation. This
    finishes one iteration of the agent–environment interaction. The user provides the model that
    helps select the actions to be executed. Additionally, the user can configure the optimizer
    that trains the model, given as a `torch.optim.Optimizer` object, as well as the corresponding
    loss function.

    :ivar _environment: A `GraphEnvironment` object that represents the RL environment whose
        graph-building game should be used to construct all the graphs.
    :ivar _new_candidates_count: A positive `int` that determines how many graphs should be
        constructed in each iteration by running the corresponding number of episodes in parallel.
    :ivar _elite_count: A positive `int` that determines how many executed episodes with the
        greatest cumulative reward should be used to train the action prediction model in each
        iteration.
    :ivar _survivors_count: A positive `int` that determines how many executed episodes with the
        greatest cumulative reward should be carried over to the next generation in each iteration.
    :ivar _policy_network: A `torch.nn.Module` object that represents the action prediction model
        that is used to compute the probability for each of the actions to be selected in each
        episode and in each step.
    :ivar _optimizer: 
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
        Docstring for __init__
        
        :param self: Description
        :param environment: Description
        :type environment: GraphEnvironment
        :param policy_network: Description
        :type policy_network: nn.Module
        :param optimizer: Description
        :type optimizer: torch.optim.Optimizer
        :param loss_function: Description
        :type loss_function: Callable
        :param new_candidates_count: Description
        :type new_candidates_count: int
        :param elite_count: Description
        :type elite_count: int
        :param survivors_count: Description
        :type survivors_count: int
        :param random_action_mechanism: Description
        :type random_action_mechanism: RandomActionMechanism
        :param rng: Description
        :type rng: Optional[np.random.Generator]
        """

        self._environment: GraphEnvironment = environment
        self._new_candidates_count: int = new_candidates_count
        self._elite_count: int = elite_count
        self._survivors_count: int = survivors_count

        self._policy_network: nn.Module = policy_network
        self._optimizer: torch.optim.Optimizer = optimizer
        self._loss_function: Callable = loss_function
        self._device: torch.device = next(self._policy_network.parameters()).device
        self._random_action_mechanism: RandomActionMechanism = random_action_mechanism

        if rng is None:
            rng = np.random.default_rng()
        self._rng: np.random.Generator = rng

        self._step_count: Optional[int] = None
        self._best_score: Optional[float] = None
        self._is_best_score_improved: Optional[bool] = None
        self._population_states: Optional[np.ndarray] = None
        self._population_actions: Optional[np.ndarray] = None
        self._population_rewards: Optional[np.ndarray] = None

    def reset(self):
        self._step_count = 0
        self._best_score = None
        self._is_best_score_improved = True

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

    def step(self):
        # if self._step_count >= 1:
        #     self._environment.initial_graph_generator = create_fixed_graph_generator(
        #         fixed_graph=self.best_graph,
        #         graph_format=GraphFormat.FLATTENED_ROW_MAJOR_BINARY,
        #     )

        state_batch, status = self._environment.reset_batch(batch_size=self._new_candidates_count)
        self._population_states[0, self._survivors_count :, :] = state_batch
        self._population_rewards[self._survivors_count :] = 0

        action_count = 0
        random_action_probability = self._random_action_mechanism(self._is_best_score_improved)

        while status == EpisodeStatus.IN_PROGRESS:
            state_batch_torch = torch.from_numpy(state_batch.astype(np.float32)).to(self._device)
            logits_batch_torch = self._policy_network(state_batch_torch)

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

            self._population_actions[action_count, self._survivors_count :] = action_batch
            state_batch, reward_batch, status = self._environment.step_batch(action_batch)

            self._population_rewards[self._survivors_count :] += reward_batch
            action_count += 1
            self._population_states[action_count, self._survivors_count :, :] = state_batch

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
            assert np.all(survivors_mask[:self._survivors_count] == False)

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
