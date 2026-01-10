import numpy as np

from rl_graph_theory.environments.graph_environment import EpisodeStatus
from rl_graph_theory.graphs.graph import FlattenedOrdering

from ..utils import batchify, insert, remove_at, replace
from .global_set_test_cases import TEST_CASES_CONSTRUCTOR as gstc_constructor
from .global_set_test_cases import TEST_CASES_RESET_BATCH as gstc_reset_batch
from .global_set_test_cases import (
    TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH as gstc_state_batch_to_graph_batch,
)

DTYPE_STATE = np.uint8
DTYPE_ACTION = np.int32

TEST_CASES_CONSTRUCTOR = insert(
    remove_at(gstc_constructor, 5),
    4,
    False,
) + insert(
    remove_at(gstc_constructor, 5),
    4,
    True,
)


TEST_CASES_RESET_BATCH = remove_at(
    [test_case for test_case in gstc_reset_batch if test_case[3] == 2], 3
)


TEST_CASES_TRANSITION_BATCH_ROW_MAJOR = [
    (
        1,
        2,
        2,
        False,
        FlattenedOrdering.ROW_MAJOR,
        False,
        False,
        0,
        np.asarray([[0]], dtype=DTYPE_STATE),
        np.asarray([[0, 0]], dtype=DTYPE_ACTION),
        np.asarray([[0]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        2,
        False,
        FlattenedOrdering.ROW_MAJOR,
        False,
        False,
        0,
        np.asarray([[0]], dtype=DTYPE_STATE),
        np.asarray([[0, 1]], dtype=DTYPE_ACTION),
        np.asarray([[1]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        2,
        False,
        FlattenedOrdering.ROW_MAJOR,
        False,
        False,
        0,
        np.asarray([[1]], dtype=DTYPE_STATE),
        np.asarray([[0, 0]], dtype=DTYPE_ACTION),
        np.asarray([[1]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        2,
        False,
        FlattenedOrdering.ROW_MAJOR,
        False,
        False,
        0,
        np.asarray([[1]], dtype=DTYPE_STATE),
        np.asarray([[0, 1]], dtype=DTYPE_ACTION),
        np.asarray([[0]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        2,
        True,
        FlattenedOrdering.ROW_MAJOR,
        False,
        False,
        0,
        np.asarray([[0]], dtype=DTYPE_STATE),
        np.asarray([[0]], dtype=DTYPE_ACTION),
        np.asarray([[1]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        2,
        True,
        FlattenedOrdering.ROW_MAJOR,
        False,
        False,
        0,
        np.asarray([[1]], dtype=DTYPE_STATE),
        np.asarray([[0]], dtype=DTYPE_ACTION),
        np.asarray([[0]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        2,
        True,
        FlattenedOrdering.ROW_MAJOR,
        True,
        False,
        0,
        np.asarray([[0, 0]], dtype=DTYPE_STATE),
        np.asarray([[0]], dtype=DTYPE_ACTION),
        np.asarray([[1, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        2,
        True,
        FlattenedOrdering.ROW_MAJOR,
        True,
        False,
        0,
        np.asarray([[0, 0]], dtype=DTYPE_STATE),
        np.asarray([[1]], dtype=DTYPE_ACTION),
        np.asarray([[0, 1]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        2,
        True,
        FlattenedOrdering.ROW_MAJOR,
        False,
        True,
        0,
        np.asarray([[0, 0, 0]], dtype=DTYPE_STATE),
        np.asarray([[1]], dtype=DTYPE_ACTION),
        np.asarray([[0, 1, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        2,
        True,
        FlattenedOrdering.ROW_MAJOR,
        False,
        True,
        0,
        np.asarray([[0, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[1]], dtype=DTYPE_ACTION),
        np.asarray([[0, 0, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        2,
        True,
        FlattenedOrdering.ROW_MAJOR,
        True,
        True,
        0,
        np.asarray([[0, 1, 0, 0]], dtype=DTYPE_STATE),
        np.asarray([[2]], dtype=DTYPE_ACTION),
        np.asarray([[0, 1, 1, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        2,
        True,
        FlattenedOrdering.ROW_MAJOR,
        True,
        True,
        1,
        np.asarray([[0, 1, 0, 0]], dtype=DTYPE_STATE),
        np.asarray([[2]], dtype=DTYPE_ACTION),
        np.asarray([[0, 1, 1, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.TRUNCATED,
    ),
]

TEST_CASES_TRANSITION_BATCH_ROW_MAJOR += [
    *batchify(
        TEST_CASES_TRANSITION_BATCH_ROW_MAJOR,
        batch_size=2,
        expand_dims=False,
    ),
    (
        2,
        2,
        2,
        True,
        FlattenedOrdering.ROW_MAJOR,
        True,
        True,
        1,
        np.asarray([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[2], [2]], dtype=DTYPE_STATE),
        np.asarray([[0, 1, 1, 0], [0, 0, 0, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.TRUNCATED,
    ),
    (
        2,
        2,
        2,
        True,
        FlattenedOrdering.ROW_MAJOR,
        True,
        True,
        1,
        np.asarray([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[3], [0]], dtype=DTYPE_STATE),
        np.asarray([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.TRUNCATED,
    ),
    (
        2,
        2,
        2,
        False,
        FlattenedOrdering.ROW_MAJOR,
        True,
        True,
        1,
        np.asarray([[0, 0, 1, 0], [0, 0, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[2, 0], [2, 1]], dtype=DTYPE_STATE),
        np.asarray([[0, 0, 1, 0], [0, 0, 0, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.TRUNCATED,
    ),
]

TEST_CASES_TRANSITION_BATCH_CLOCKWISE = replace(
    TEST_CASES_TRANSITION_BATCH_ROW_MAJOR,
    FlattenedOrdering.ROW_MAJOR,
    FlattenedOrdering.CLOCKWISE,
)

TEST_CASES_TRANSITION_BATCH = (
    TEST_CASES_TRANSITION_BATCH_ROW_MAJOR + TEST_CASES_TRANSITION_BATCH_CLOCKWISE
)


TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH = remove_at(
    [test_case for test_case in gstc_state_batch_to_graph_batch if test_case[3] == 2], 3
)
