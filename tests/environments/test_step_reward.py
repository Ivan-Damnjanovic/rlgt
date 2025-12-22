import numpy as np
import pytest
from rl_graph_theory.environments.graph_environment import (
    GraphEnvironment,
    RewardType,
    EpisodeStatus,
)


class StepRewardTestingEnvironment(GraphEnvironment):
    transition_function = None

    def reset_batch(self, batch_size):
        self._status = EpisodeStatus.IN_PROGRESS

    def _transition_batch(self, action_batch):
        if self.transition_function is None:
            return

        self.transition_function(self, action_batch)

    def state_batch_to_graph_batch(self, state_batch):
        return state_batch


def test_not_in_progress():
    env = StepRewardTestingEnvironment(RewardType.SPARSE, lambda _: None)
    env.reset_batch(1)

    env._status = EpisodeStatus.TERMINATED

    with pytest.raises(RuntimeError):
        env.step_batch(None)


@pytest.mark.parametrize(
    "batch_size, state_batch, status",
    [
        (
            1,
            np.asarray([[0, 0, 0, 1, 0, 0]], dtype=int),
            EpisodeStatus.IN_PROGRESS,
        ),
        (
            1,
            np.asarray([[0, 0, 0, 1, 0, 0]], dtype=int),
            EpisodeStatus.TERMINATED,
        ),
        (
            3,
            np.asarray([[0, 0, 0, 1, 0, 0]], dtype=int),
            EpisodeStatus.IN_PROGRESS,
        ),
        (
            4,
            np.asarray([[0, 0, 0, 1, 0, 0]], dtype=int),
            EpisodeStatus.TERMINATED,
        ),
    ],
)
def test_sparese(batch_size, state_batch, status):
    state_batch = np.stack([state_batch] * batch_size)

    def r(graph_batch):
        assert status is EpisodeStatus.TERMINATED
        np.testing.assert_array_equal(graph_batch, state_batch)
        return np.ones((batch_size,))

    def t(self: StepRewardTestingEnvironment, action_batch):
        self._state_batch = state_batch
        self._status = status

    env = StepRewardTestingEnvironment(RewardType.SPARSE, r)
    env.reset_batch(batch_size)
    env.transition_function = t

    _, calculated_reward, _ = env.step_batch(None)

    np.testing.assert_array_equal(
        calculated_reward,
        np.full(
            (batch_size,),
            int(status is EpisodeStatus.TERMINATED),
        ),
    )


@pytest.mark.parametrize(
    "batch_size, state_batch",
    [
        (1, np.asarray([[0, 0, 0, 1, 0, 0]], dtype=int)),
        (4, np.asarray([[0, 0, 0, 1, 0, 0]], dtype=int)),
    ],
)
def test_telesopic(batch_size, state_batch):
    state_batch = np.stack([state_batch] * batch_size)

    def r(graph_batch, c=[0]):
        c[0] += 1  # Carried over since list definition is within the function definition.

        assert c[0] in [1, 2]  # Only two calls.

        if c[0] == 1:  # First call, state after transition.
            np.testing.assert_array_equal(graph_batch, state_batch)  # State set during transition.
        elif c[0] == 2:  # Second call, state before transition.
            np.testing.assert_array_equal(graph_batch, None)  # Initial state is None.

        return np.full((batch_size,), c[0])  # Reward is 1 on the first call, and 2 on the second.

    def t(self: StepRewardTestingEnvironment, action_batch):
        self._state_batch = state_batch

    env = StepRewardTestingEnvironment(RewardType.TELESCOPIC, r)
    env.reset_batch(batch_size)
    env.transition_function = t

    _, calculated_reward, _ = env.step_batch(None)

    # Reward function returns 1 for the new state and 2 for the old state,
    # difference is always -1.
    np.testing.assert_array_equal(calculated_reward, np.full((batch_size,), -1))


@pytest.mark.parametrize(
    "batch_size, state_batch",
    [
        (1, np.asarray([[0, 0, 0, 1, 0, 0]], dtype=int)),
        (4, np.asarray([[0, 0, 0, 1, 0, 0]], dtype=int)),
    ],
)
def test_proper(batch_size, state_batch):
    state_batch = np.stack([state_batch] * batch_size)

    def r(old_graph_batch, new_graph_batch):
        np.testing.assert_array_equal(old_graph_batch, None)
        np.testing.assert_array_equal(new_graph_batch, state_batch)
        return np.ones((batch_size,))

    def t(self: StepRewardTestingEnvironment, action_batch):
        self._state_batch = state_batch

    env = StepRewardTestingEnvironment(RewardType.PROPER, r)
    env.reset_batch(batch_size)
    env.transition_function = t

    _, calculated_reward, _ = env.step_batch(None)

    np.testing.assert_array_equal(calculated_reward, np.ones((batch_size,)))
