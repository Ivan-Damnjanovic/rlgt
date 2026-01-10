import numpy as np
import pytest

from rl_graph_theory.environments.local_environments import (
    EpisodeStatus,
    FlattenedOrdering,
    LocalSetEnvironment,
    RewardType,
)

from .local_set_test_cases import (
    TEST_CASES_CONSTRUCTOR,
    TEST_CASES_RESET_BATCH,
    TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH,
    TEST_CASES_TRANSITION_BATCH,
)


@pytest.mark.parametrize(
    "reward_type, reward_function, graph_order, episode_length, flattened_ordering, edge_colors, "
    "is_directed, allow_loops, initial_graph_generator, expected_flattened_length",
    TEST_CASES_CONSTRUCTOR,
)
def test_constructor(
    reward_type,
    reward_function,
    graph_order,
    episode_length,
    flattened_ordering,
    edge_colors,
    is_directed,
    allow_loops,
    initial_graph_generator,
    expected_flattened_length,
):
    env = LocalSetEnvironment(
        reward_type,
        reward_function,
        graph_order,
        episode_length,
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

    assert env._current_vertices is None
    assert env._step_count is None

    assert (
        env.episode_length == expected_flattened_length
        if episode_length is None
        else episode_length
    )


@pytest.mark.parametrize(
    "batch_size, graph_order, flattened_ordering, edge_colors, is_directed, allow_loops, "
    "expected_state",
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
):
    env = LocalSetEnvironment(
        RewardType.PROPER,
        lambda _: np.empty(0),
        graph_order,
        None,
        flattened_ordering,
        edge_colors,
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
    "batch_size, graph_order, episode_length, flattened_ordering, edge_colors, is_directed, allow_loops, "
    "next_index, init_state, action_batch, state_batch, status",
    TEST_CASES_TRANSITION_BATCH,
)
def test_transition_batch(
    batch_size,
    graph_order,
    episode_length,
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
    env = LocalSetEnvironment(
        RewardType.PROPER,
        lambda _: np.empty(0),
        graph_order,
        episode_length,
        flattened_ordering,
        edge_colors,
        is_directed,
        allow_loops,
    )

    _ = env.reset_batch(batch_size)

    env._current_vertices = np.argmax(init_state[:, -graph_order:], axis=1)
    env._state_batch = init_state
    env._step_count = next_index

    env._transition_batch(action_batch[:, 0] + action_batch[:, 1] * graph_order)

    np.testing.assert_array_equal(env._state_batch, state_batch)

    assert env._status is status


def test_transition_batch_runtime_error():
    env = LocalSetEnvironment(
        RewardType.PROPER,
        lambda _: np.empty(0),
        graph_order=2,
        episode_length=None,
        flattened_ordering=FlattenedOrdering.ROW_MAJOR,
        edge_colors=2,
        is_directed=False,
        allow_loops=False,
    )

    _ = env.reset_batch(1)

    with pytest.raises(RuntimeError):
        env._transition_batch(np.asarray([[0, 1]], dtype=int))


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
    env = LocalSetEnvironment(
        RewardType.PROPER,
        lambda _: np.empty(0),
        graph_order,
        None,
        flattened_ordering,
        edge_colors,
        is_directed,
        allow_loops,
    )

    graph_batch = env.state_batch_to_graph_batch(state_batch)
    np.testing.assert_array_equal(
        (
            graph_batch.flattened_clockwise_colors
            if flattened_ordering is FlattenedOrdering.CLOCKWISE
            else graph_batch.flattened_row_major_colors
        ),
        flattened,
    )


def test_limit():
    env = LocalSetEnvironment(
        RewardType.TELESCOPIC,
        lambda a: np.sum(a.flattened_row_major_colors, axis=1),
        graph_order=2,
        flattened_ordering=FlattenedOrdering.ROW_MAJOR,
        edge_colors=255,
        is_directed=False,
        allow_loops=False,
    )

    env.reset_batch(1)
    state, reward, status = env.step_batch(np.asarray([1 + 254 * 2], np.int32))

    print(state, reward, status)

    np.testing.assert_array_equal(state, [[0] * 253 + [1, 0, 1]])
    np.testing.assert_array_equal(reward, [254])
    assert status is EpisodeStatus.TRUNCATED


@pytest.mark.parametrize(
    "env, mask",
    [
        (
            LocalSetEnvironment(RewardType.PROPER, lambda _: np.empty(0), 2, allow_loops=True),
            None,
        ),
        (
            LocalSetEnvironment(RewardType.PROPER, lambda _: np.empty(0), 2),
            np.asarray([[False, True, False, True]], np.bool_),
        ),
        (
            LocalSetEnvironment(RewardType.PROPER, lambda _: np.empty(0), 3),
            np.asarray([[False, True, True, False, True, True]], np.bool_),
        ),
    ],
)
def test_action_mask(env: LocalSetEnvironment, mask: np.ndarray | None):
    assert env.action_mask is None

    env.reset_batch(1)

    if mask is None:
        assert env.action_mask is None
    else:
        np.testing.assert_array_equal(env.action_mask, mask)
