import numpy as np
import pytest

from rlgt.environments.graph_environment import EpisodeStatus, GraphEnvironment


class StepTestingEnvironment(GraphEnvironment):
    transition_function = None

    def _initialize_batch(self, batch_size):
        self._state_batch = np.zeros((batch_size, 1), np.uint8)
        self._status = EpisodeStatus.IN_PROGRESS

    def _transition_batch(self, action_batch):
        if self.transition_function is None:
            return

        self.transition_function(self, action_batch)

    def state_batch_to_graph_batch(self, state_batch):
        return state_batch.copy()

    def action_mask():
        return

    def action_number():
        return

    def episode_length():
        return

    def state_dtype():
        return

    def state_length():
        return

    def is_continuing():
        return


def test_not_in_progress():
    env = StepTestingEnvironment(lambda _: None, sparse_setting=True)
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

    def t(self: StepTestingEnvironment, action_batch):
        self._state_batch = state_batch
        self._status = status

    env = StepTestingEnvironment(r, sparse_setting=True)
    env.reset_batch(batch_size)
    env.transition_function = t

    _, calculated_reward, _ = env.step_batch(None)

    if status is EpisodeStatus.TERMINATED:
        np.testing.assert_array_equal(calculated_reward, np.full((batch_size,), 1))
    else:
        assert calculated_reward is None


@pytest.mark.parametrize(
    "batch_size, state_batch_value",
    [(1, 1), (4, 1)],
)
def test_telesopic(batch_size, state_batch_value):
    def r(graph_batch, c=[0]):
        c[0] += 1  # Carried over since list definition is within the function definition.

        assert c[0] in [1, 2]  # Only two calls.

        if c[0] == 1:  # First call, state after transition.
            assert np.all(graph_batch == 1)  # State set during transition.
        elif c[0] == 2:  # Second call, state before transition.
            assert np.all(graph_batch == 0)  # Initial state is None.

        return np.full((batch_size,), c[0])  # Reward is 1 on the first call, and 2 on the second.

    def t(self: StepTestingEnvironment, action_batch):
        self._state_batch[...] = state_batch_value

    env = StepTestingEnvironment(lambda _: 0, r)
    env.reset_batch(batch_size)
    env.transition_function = t

    _, calculated_reward, _ = env.step_batch(None)

    # Reward function returns 1 for the new state and 2 for the old state,
    # difference is always -1.
    np.testing.assert_array_equal(calculated_reward, np.full((batch_size,), 2))


@pytest.mark.parametrize(
    "batch_size, state_batch_value",
    [(1, 1), (4, 1)],
)
def test_proper(batch_size, state_batch_value):
    def r(old_graph_batch, new_graph_batch):
        assert np.all(new_graph_batch == 1)
        assert np.all(old_graph_batch == 0)
        return np.ones((batch_size,))

    def t(self: StepTestingEnvironment, action_batch):
        self._state_batch[...] = state_batch_value

    env = StepTestingEnvironment(lambda _: 0, r)
    env.reset_batch(batch_size)
    env.transition_function = t

    _, calculated_reward, _ = env.step_batch(None)

    np.testing.assert_array_equal(calculated_reward, np.ones((batch_size,)))
