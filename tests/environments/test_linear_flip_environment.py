import numpy as np
import pytest

from rlgt.environments.linear_environments import (
    EpisodeStatus,
    FlattenedOrdering,
    LinearFlipEnvironment,
)

from .linear_flip_test_cases import (
    TEST_CASES_CONSTRUCTOR,
    TEST_CASES_RESET_BATCH,
    TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH,
    TEST_CASES_TRANSITION_BATCH,
)


@pytest.mark.parametrize(
    "reward_function, graph_order, flattened_ordering, "
    "is_directed, allow_loops, initial_graph_generator, expected_flattened_length",
    TEST_CASES_CONSTRUCTOR,
)
def test_constructor(
    reward_function,
    graph_order,
    flattened_ordering,
    is_directed,
    allow_loops,
    initial_graph_generator,
    expected_flattened_length,
):
    env = LinearFlipEnvironment(
        reward_function,
        graph_order,
        flattened_ordering,
        is_directed,
        allow_loops,
        initial_graph_generator,
    )

    assert getattr(env, "__GraphEnvironment_reward_function", reward_function)

    assert env._is_directed == is_directed
    assert env._allow_loops == allow_loops
    assert env._flattened_ordering == flattened_ordering

    if initial_graph_generator is not None:
        assert env.initial_graph_generator == initial_graph_generator

    assert env._flattened_length == expected_flattened_length

    assert env._step_count is None


@pytest.mark.parametrize(
    "batch_size, graph_order, flattened_ordering, is_directed, allow_loops, "
    "expected_state, initial_graph_generator",
    TEST_CASES_RESET_BATCH,
)
def test_reset_batch(
    batch_size,
    graph_order,
    flattened_ordering,
    is_directed,
    allow_loops,
    expected_state,
    initial_graph_generator,
):
    env = LinearFlipEnvironment(
        lambda _: np.empty(0),
        graph_order,
        flattened_ordering,
        is_directed,
        allow_loops,
        initial_graph_generator,
    )

    state_batch, _, status = env.reset_batch(batch_size)

    assert env._step_count == 0
    assert status is env._status is EpisodeStatus.IN_PROGRESS

    np.testing.assert_array_equal(state_batch, env._state_batch)
    np.testing.assert_array_equal(state_batch, expected_state)

    assert state_batch.shape[1] == env.state_length
    assert state_batch.dtype.type is env.state_dtype


@pytest.mark.parametrize(
    "batch_size, graph_order, flattened_ordering, is_directed, allow_loops, "
    "next_index, init_state, action_batch, state_batch, status",
    TEST_CASES_TRANSITION_BATCH[:6],
)
def test_transition_batch(
    batch_size,
    graph_order,
    flattened_ordering,
    is_directed,
    allow_loops,
    next_index,
    init_state,
    action_batch,
    state_batch,
    status,
):
    env = LinearFlipEnvironment(
        lambda _: np.empty(0),
        graph_order,
        flattened_ordering,
        is_directed,
        allow_loops,
    )

    _ = env.reset_batch(batch_size)

    env._state_batch = init_state
    env._step_count = next_index

    env._transition_batch(action_batch.flatten())

    np.testing.assert_array_equal(env._state_batch, state_batch)

    assert env._status is status


@pytest.mark.parametrize(
    "batch_size, graph_order, flattened_ordering, is_directed, allow_loops, state_batch, flattened",
    TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH,
)
def test_state_batch_to_graph_batch(
    batch_size,
    graph_order,
    flattened_ordering,
    is_directed,
    allow_loops,
    state_batch,
    flattened,
):
    env = LinearFlipEnvironment(
        lambda _: np.empty(0),
        graph_order,
        flattened_ordering,
        is_directed,
        allow_loops,
    )

    graph_batch = env.state_batch_to_graph_batch(state_batch)
    np.testing.assert_array_equal(
        flattened,
        (
            graph_batch.flattened_clockwise_colors
            if flattened_ordering is FlattenedOrdering.CLOCKWISE
            else graph_batch.flattened_row_major_colors
        ),
    )
