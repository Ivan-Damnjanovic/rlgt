import numpy as np
import pytest

from rl_graph_theory.environments.global_environments import (
    EpisodeStatus,
    FlattenedOrdering,
    GlobalFlipEnvironment,
    RewardType,
)

from .global_flip_test_cases import (
    TEST_CASES_CONSTRUCTOR,
    TEST_CASES_RESET_BATCH,
    TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH,
    TEST_CASES_TRANSITION_BATCH,
)


@pytest.mark.parametrize(
    "reward_type, reward_function, graph_order, episode_length, flip_only, flattened_ordering, "
    "is_directed, allow_loops, initial_graph_generator, expected_flattened_length",
    TEST_CASES_CONSTRUCTOR,
)
def test_constructor(
    reward_type,
    reward_function,
    graph_order,
    episode_length,
    flip_only,
    flattened_ordering,
    is_directed,
    allow_loops,
    initial_graph_generator,
    expected_flattened_length,
):
    env = GlobalFlipEnvironment(
        reward_type,
        reward_function,
        graph_order,
        episode_length,
        flip_only,
        flattened_ordering,
        is_directed,
        allow_loops,
        initial_graph_generator,
    )

    assert getattr(env, "__GraphEnvironment_reward_type", reward_type)
    assert getattr(env, "__GraphEnvironment_reward_function", reward_function)

    assert env._is_directed == is_directed
    assert env._allow_loops == allow_loops
    assert env._flattened_ordering == flattened_ordering

    if initial_graph_generator is not None:
        assert env.initial_graph_generator == initial_graph_generator

    assert env._flattened_length == expected_flattened_length

    assert env._step_count is None

    assert (
        env.episode_length == expected_flattened_length
        if episode_length is None
        else episode_length
    )


@pytest.mark.parametrize(
    "batch_size, graph_order, flattened_ordering, is_directed, allow_loops, expected_state",
    TEST_CASES_RESET_BATCH,
)
def test_reset_batch(
    batch_size,
    graph_order,
    flattened_ordering,
    is_directed,
    allow_loops,
    expected_state,
):
    env = GlobalFlipEnvironment(
        RewardType.PROPER,
        lambda _: np.empty(0),
        graph_order,
        None,
        False,
        flattened_ordering,
        is_directed,
        allow_loops,
    )

    state_batch, status = env.reset_batch(batch_size)

    assert env._step_count == 0
    assert status is env._status is EpisodeStatus.IN_PROGRESS

    np.testing.assert_array_equal(state_batch, env._state_batch)
    np.testing.assert_array_equal(state_batch, expected_state)

    assert state_batch.shape[1] == env.state_length
    assert state_batch.dtype.type is env.state_dtype


@pytest.mark.parametrize(
    "batch_size, graph_order, episode_length, flip_only, flattened_ordering, is_directed, allow_loops, "
    "next_index, init_state, action_batch, state_batch, status",
    TEST_CASES_TRANSITION_BATCH,
)
def test_transition_batch(
    batch_size,
    graph_order,
    episode_length,
    flip_only,
    flattened_ordering,
    is_directed,
    allow_loops,
    next_index,
    init_state,
    action_batch,
    state_batch,
    status,
):
    env = GlobalFlipEnvironment(
        RewardType.PROPER,
        lambda _: np.empty(0),
        graph_order,
        episode_length,
        flip_only,
        flattened_ordering,
        is_directed,
        allow_loops,
    )

    _ = env.reset_batch(batch_size)

    env._state_batch = init_state.copy()
    env._step_count = next_index

    if flip_only:
        env._transition_batch(action_batch.flatten())
    else:
        env._transition_batch(action_batch[:, 0] + action_batch[:, 1] * env._flattened_length)

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
    env = GlobalFlipEnvironment(
        RewardType.PROPER,
        lambda _: np.empty(0),
        graph_order,
        None,
        False,
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
