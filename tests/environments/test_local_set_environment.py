import numpy as np
import pytest

from rl_graph_theory.environments.local_environments import (
    EpisodeStatus,
    FlattenedOrdering,
    LocalSetEnvironment,
    RewardType,
)

from .global_set_test_cases import (
    TEST_CASES_CONSTRUCTOR,
    # TEST_CASES_RESET_BATCH,
    # TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH,
    # TEST_CASES_TRANSITION_BATCH,
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
