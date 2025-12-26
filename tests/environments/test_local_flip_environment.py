import numpy as np
import pytest

from rl_graph_theory.environments.local_environments import (
    EpisodeStatus,
    FlattenedOrdering,
    LocalFlipEnvironment,
    RewardType,
)

from .local_flip_test_cases import (
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
    env = LocalFlipEnvironment(
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

    assert env._flip_only == flip_only
    assert env._is_directed == is_directed
    assert env._allow_loops == allow_loops
    assert env._flattened_ordering == flattened_ordering

    if initial_graph_generator is not None:
        assert env.initial_graph_generator == initial_graph_generator

    assert env._flattened_length == expected_flattened_length

    assert env._current_vertices is None
    assert env._step_count is None

    assert (
        env.episode_length == expected_flattened_length
        if episode_length is None
        else episode_length
    )


@pytest.mark.parametrize(
    "batch_size, graph_order, flattened_ordering, is_directed, allow_loops, "
    "expected_state",
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
    env = LocalFlipEnvironment(
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
    env = LocalFlipEnvironment(
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

    env._current_vertices = np.argmax(init_state[:, -graph_order:], axis=1)
    env._state_batch = init_state
    env._step_count = next_index

    env._transition_batch(action_batch)

    np.testing.assert_array_equal(env._state_batch, state_batch)

    assert env._status is status


def test_transition_batch_runtime_error():
    env = LocalFlipEnvironment(
        RewardType.PROPER,
        lambda _: np.empty(0),
        graph_order=2,
        episode_length=None,
        flattened_ordering=FlattenedOrdering.ROW_MAJOR,
        is_directed=False,
        allow_loops=False,
    )

    _ = env.reset_batch(1)

    with pytest.raises(RuntimeError):
        env._transition_batch(np.asarray([[0, 1]], dtype=int))


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
    env = LocalFlipEnvironment(
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
            graph_batch.flattened_clockwise
            if flattened_ordering is FlattenedOrdering.CLOCKWISE
            else graph_batch.flattened_row_major
        ),
    )
