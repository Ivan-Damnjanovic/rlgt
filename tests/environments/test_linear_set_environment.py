import numpy as np
import pytest

from rl_graph_theory.environments.linear_environments import (
    EpisodeStatus,
    FlattenedOrdering,
    LinearSetEnvironment,
    RewardType,
)

from .linear_set_test_cases import (
    TEST_CASES_CONSTRUCTOR,
    TEST_CASES_RESET_BATCH,
    TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH,
    TEST_CASES_TRANSITION_BATCH,
)


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


@pytest.mark.parametrize(
    "batch_size, graph_order, flattened_ordering, edge_colors, is_directed, allow_loops, "
    "next_index, init_state, action_batch, state_batch, status",
    TEST_CASES_TRANSITION_BATCH,
)
def test_transition_batch(
    batch_size,
    graph_order,
    flattened_ordering,
    edge_colors,
    is_directed,
    allow_loops,
    next_index,
    init_state,
    action_batch,
    state_batch,
    status,
):
    env = LinearSetEnvironment(
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
    env._step_count = next_index

    env._transition_batch(action_batch)

    np.testing.assert_array_equal(env._state_batch, state_batch)

    assert env._status is status


@pytest.mark.parametrize(
    "batch_size, graph_order, flattened_ordering, edge_colors, is_directed, allow_loops, state_batch, flattened",
    TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH,
)
def test_state_batch_to_graph_batch(
    batch_size,
    graph_order,
    flattened_ordering,
    edge_colors,
    is_directed,
    allow_loops,
    state_batch,
    flattened,
):
    env = LinearSetEnvironment(
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


def test_limit():
    env = LinearSetEnvironment(
        RewardType.TELESCOPIC,
        lambda a: np.sum(a.flattened_row_major, axis=1),
        graph_order=2,
        flattened_ordering=FlattenedOrdering.ROW_MAJOR,
        edge_colors=255,
        is_directed=False,
        allow_loops=False,
    )

    env.reset_batch(1)
    state, reward, status = env.step_batch(np.asarray([[254]], np.uint8))

    np.testing.assert_array_equal(state, [[0] * 253 + [1, 0]])
    np.testing.assert_array_equal(reward, [254])
    assert status is EpisodeStatus.TERMINATED
