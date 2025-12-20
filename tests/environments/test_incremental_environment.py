import numpy as np
import pytest

from rl_graph_theory.environments.incremental_environment import (
    EpisodeStatus,
    IncrementalEnvironment,
    RewardType,
)
from rl_graph_theory.graphs.graph import FlattenedOrdering

from .incremental_test_cases import (
    TEST_CASES_CONSTRUCTOR,
    TEST_CASES_RESET_BATCH,
    TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH,
    TEST_CASES_TRANSITION_BATCH,
)


@pytest.mark.parametrize(
    "reward_type, reward_function, graph_order, flattened_ordering, "
    "edge_colors, is_directed, allow_loops, expected_flattened_length",
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
    expected_flattened_length,
):
    env = IncrementalEnvironment(
        reward_type,
        reward_function,
        graph_order,
        flattened_ordering,
        edge_colors,
        is_directed,
        allow_loops,
    )

    assert getattr(env, "__GraphEnvironment_reward_type", reward_type)
    assert getattr(env, "__GraphEnvironment_reward_function", reward_function)

    assert env._state_batch is None
    assert env._status is None

    assert env._edge_colors == edge_colors
    assert env._is_directed == is_directed
    assert env._allow_loops == allow_loops
    assert env._flattened_ordering == flattened_ordering

    assert env._flattened_length == expected_flattened_length

    assert env._next_entry_index is None


@pytest.mark.parametrize(
    "graph_order, flattened_ordering, edge_colors, is_directed, allow_loops, batch_size, expected_state",
    TEST_CASES_RESET_BATCH,
)
def test_reset_batch(
    graph_order,
    flattened_ordering,
    edge_colors,
    is_directed,
    allow_loops,
    batch_size,
    expected_state,
):
    env = IncrementalEnvironment(
        RewardType.PROPER,
        lambda _: np.empty(0),
        graph_order,
        flattened_ordering,
        edge_colors,
        is_directed,
        allow_loops,
    )

    state_batch, status = env.reset_batch(batch_size)

    assert env._next_entry_index == 0
    assert status is env._status is EpisodeStatus.IN_PROGRESS

    np.testing.assert_array_equal(state_batch, env._state_batch)
    np.testing.assert_array_equal(state_batch, expected_state)


@pytest.mark.parametrize(
    "graph_order, flattened_ordering, edge_colors, is_directed, allow_loops, batch_size, "
    "next_index, init_state, action_batch, state_batch, status",
    TEST_CASES_TRANSITION_BATCH,
)
def test_transition_batch(
    graph_order,
    flattened_ordering,
    edge_colors,
    is_directed,
    allow_loops,
    batch_size,
    next_index,
    init_state,
    action_batch,
    state_batch,
    status,
):
    env = IncrementalEnvironment(
        RewardType.PROPER,
        lambda _: np.empty(0),
        graph_order,
        flattened_ordering,
        edge_colors,
        is_directed,
        allow_loops,
    )

    _ = env.reset_batch(batch_size)

    env._state_batch = init_state
    env._next_entry_index = next_index

    env._transition_batch(action_batch)

    np.testing.assert_array_equal(env._state_batch, state_batch)

    assert env._status is status


@pytest.mark.parametrize(
    "graph_order, flattened_ordering, edge_colors, is_directed, allow_loops, batch_size, state_batch, flattened",
    TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH,
)
def test_state_batch_to_graph_batch(
    graph_order,
    flattened_ordering,
    edge_colors,
    is_directed,
    allow_loops,
    batch_size,
    state_batch,
    flattened,
):
    env = IncrementalEnvironment(
        RewardType.PROPER,
        lambda _: np.empty(0),
        graph_order,
        flattened_ordering,
        edge_colors,
        is_directed,
        allow_loops,
    )

    graph_batch = env.state_batch_to_graph_batch(state_batch)
    np.testing.assert_array_equal(
        flattened,
        (
            graph_batch.flattened_clockwise
            if flattened_ordering is FlattenedOrdering.CLOCKWISE
            else graph_batch.flattened_row_major
        ),
    )
