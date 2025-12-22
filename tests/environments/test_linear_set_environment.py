import pytest
import numpy as np

from rl_graph_theory.environments.linear_environments import (
    LinearSetEnvironment,
    GraphGenerator,
    RewardType,
    EpisodeStatus,
)
from .linear_set_test_cases import TEST_CASES_CONSTRUCTOR, TEST_CASES_RESET_BATCH


@pytest.mark.parametrize(
    "reward_type, reward_function, graph_order, flattened_ordering, edge_colors, "
    "is_directed, allow_loops, initial_graph_generator, expected_flattened_length",
    TEST_CASES_CONSTRUCTOR,
)
def test_constructor(
    reward_type,
    reward_function,
    graph_order,
    flattened_ordering,
    edge_colors,
    is_directed,
    allow_loops,
    initial_graph_generator,
    expected_flattened_length,
):
    env = LinearSetEnvironment(
        reward_type,
        reward_function,
        graph_order,
        flattened_ordering,
        edge_colors,
        is_directed,
        allow_loops,
        initial_graph_generator,
    )

    assert getattr(env, "__GraphEnvironment_reward_type", reward_type)
    assert getattr(env, "__GraphEnvironment_reward_function", reward_function)

    assert env._edge_colors == edge_colors
    assert env._is_directed == is_directed
    assert env._allow_loops == allow_loops
    assert env._flattened_ordering == flattened_ordering

    if initial_graph_generator is not None:
        assert env.initial_graph_generator == initial_graph_generator

    assert env._flattened_length == expected_flattened_length

    assert env._step_count is None


@pytest.mark.parametrize(
    "batch_size, graph_order, flattened_ordering, edge_colors, is_directed, allow_loops, "
    "expected_state, initial_graph_generator",
    TEST_CASES_RESET_BATCH,
)
def test_reset_batch(
    batch_size,
    graph_order,
    flattened_ordering,
    edge_colors,
    is_directed,
    allow_loops,
    expected_state,
    initial_graph_generator,
):
    env = LinearSetEnvironment(
        RewardType.PROPER,
        lambda _: np.empty(0),
        graph_order,
        flattened_ordering,
        edge_colors,
        is_directed,
        allow_loops,
        initial_graph_generator,
    )

    state_batch, status = env.reset_batch(batch_size)

    assert env._step_count == 0
    assert status is env._status is EpisodeStatus.IN_PROGRESS

    np.testing.assert_array_equal(state_batch, env._state_batch)
    np.testing.assert_array_equal(state_batch, expected_state)
