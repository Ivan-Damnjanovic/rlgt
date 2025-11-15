"""
#TODO
"""

from typing import Callable, List

import numpy as np
import pytest

from rl_graph_theory.environments.concrete_environments import IncrementalEnvironment
from rl_graph_theory.environments.environment import (
    ActionBatch,
    EpisodeStatus,
    RewardBatch,
    RewardType,
    StateBatch,
)
from rl_graph_theory.graphs.graph import EdgeOrdering

from .concrete_environments_test_cases import INCREMENTAL_ENVIRONMENT_TEST_CASES


@pytest.mark.parametrize(
    "graph_order, edge_colors, edge_ordering, reward_type, reward_function, batch_size, "
    "action_batch_history, state_batch_history, reward_batch_history, final_status, "
    "output_graph_batch_bitmask",
    INCREMENTAL_ENVIRONMENT_TEST_CASES,
)
def test_incremental_environment(
    graph_order: int,
    edge_colors: int,
    edge_ordering: EdgeOrdering,
    reward_type: RewardType,
    reward_function: Callable,
    batch_size: int,
    action_batch_history: List[ActionBatch],
    state_batch_history: List[StateBatch],
    reward_batch_history: List[RewardBatch],
    final_status: EpisodeStatus,
    output_graph_batch_bitmask: np.ndarray,
):
    environment = IncrementalEnvironment(
        graph_order=graph_order,
        edge_colors=edge_colors,
        edge_ordering=edge_ordering,
        reward_type=reward_type,
        reward_function=reward_function,
    )

    state_batch, status = environment.reset_batch(batch_size=batch_size)
    assert np.all(state_batch.data == state_batch_history[0].data)
    assert status == EpisodeStatus.IN_PROGRESS

    for index, (action_batch, correct_state_batch, correct_reward_batch) in enumerate(
        zip(action_batch_history, state_batch_history[1:], reward_batch_history)
    ):
        state_batch, reward_batch, status = environment.step_batch(action_batch=action_batch)
        assert np.all(state_batch.data == correct_state_batch.data)
        assert np.all(reward_batch.data == correct_reward_batch.data)

        if index < len(action_batch_history) - 1:
            assert status == EpisodeStatus.IN_PROGRESS
        else:
            assert status == final_status

    output_graph_batch = environment.state_batch_to_graph_batch(state_batch=state_batch)
    assert np.all(output_graph_batch.bitmask_batch == output_graph_batch_bitmask)
