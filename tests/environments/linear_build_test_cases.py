import numpy as np

from rlgt.environments.graph_environment import EpisodeStatus
from rlgt.graphs.graph import FlattenedOrdering

from ..utils import batchify, replace


DTYPE_STATE = np.uint8
DTYPE_ACTION = np.int32

TEST_CASES_CONSTRUCTOR = [
    (
        lambda _: np.empty(0),
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        1,
    ),
    (
        lambda _: np.empty(1),
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        1,
    ),
    (
        lambda _: np.empty(2),
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        1,
    ),
    (
        lambda _: np.empty(0),
        2,
        FlattenedOrdering.ROW_MAJOR,
        4,
        False,
        False,
        1,
    ),
    (
        lambda _: np.empty(0),
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        3,
    ),
    (
        lambda _: np.empty(0),
        4,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        6,
    ),
    (
        lambda _: np.empty(0),
        4,
        FlattenedOrdering.ROW_MAJOR,
        2,
        True,
        False,
        6 * 2,
    ),
    (
        lambda _: np.empty(0),
        4,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        True,
        6 + 4,
    ),
    (
        lambda _: np.empty(0),
        4,
        FlattenedOrdering.ROW_MAJOR,
        2,
        True,
        True,
        6 * 2 + 4,
    ),
    (
        lambda _: np.empty(0),
        4,
        FlattenedOrdering.CLOCKWISE,
        2,
        False,
        False,
        6,
    ),
    (
        lambda _: np.empty(0),
        4,
        FlattenedOrdering.CLOCKWISE,
        2,
        True,
        False,
        6 * 2,
    ),
    (
        lambda _: np.empty(0),
        4,
        FlattenedOrdering.CLOCKWISE,
        2,
        False,
        True,
        6 + 4,
    ),
    (
        lambda _: np.empty(0),
        4,
        FlattenedOrdering.CLOCKWISE,
        2,
        True,
        True,
        6 * 2 + 4,
    ),
]

TEST_CASES_RESET_BATCH_ROW_MAJOR = [
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        [[0, 1]],
    ),
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        3,
        False,
        False,
        [[0, 0, 1]],
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        [[0, 0, 0, 1, 0, 0]],
    ),
    (
        2,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        [[0, 1], [0, 1]],
    ),
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        True,
        False,
        [[0, 0, 1, 0]],
    ),
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        True,
        [[0, 0, 0, 1, 0, 0]],
    ),
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        True,
        True,
        [[0, 0, 0, 0, 1, 0, 0, 0]],
    ),
]

TEST_CASES_RESET_BATCH_ROW_MAJOR += [
    *batchify(
        [test_case for test_case in TEST_CASES_RESET_BATCH_ROW_MAJOR if test_case[0] == 1],
        batch_size=2,
        expand_dims=False,
    ),
]


TEST_CASES_RESET_BATCH_CLOCKWISE = replace(
    TEST_CASES_RESET_BATCH_ROW_MAJOR,
    FlattenedOrdering.ROW_MAJOR,
    FlattenedOrdering.CLOCKWISE,
)

TEST_CASES_RESET_BATCH = TEST_CASES_RESET_BATCH_ROW_MAJOR + TEST_CASES_RESET_BATCH_CLOCKWISE

TEST_CASES_TRANSITION_BATCH_ROW_MAJOR = [
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        0,
        np.asarray([[0, 1]], dtype=DTYPE_STATE),
        np.asarray([[0]], dtype=DTYPE_ACTION),
        np.asarray([[0, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.TERMINATED,
    ),
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        0,
        np.asarray([[0, 1]], dtype=DTYPE_STATE),
        np.asarray([[1]], dtype=DTYPE_ACTION),
        np.asarray([[1, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.TERMINATED,
    ),
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        3,
        False,
        False,
        0,
        np.asarray([[0, 0, 1]], dtype=DTYPE_STATE),
        np.asarray([[2]], dtype=DTYPE_ACTION),
        np.asarray([[0, 1, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.TERMINATED,
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        0,
        np.asarray([[0, 0, 0, 1, 0, 0]], dtype=DTYPE_STATE),
        np.asarray([[0]], dtype=DTYPE_ACTION),
        np.asarray([[0, 0, 0, 0, 1, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        0,
        np.asarray([[0, 0, 0, 1, 0, 0]], dtype=DTYPE_STATE),
        np.asarray([[1]], dtype=DTYPE_ACTION),
        np.asarray([[1, 0, 0, 0, 1, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        1,
        np.asarray([[0, 0, 0, 0, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[1]], dtype=DTYPE_ACTION),
        np.asarray([[0, 1, 0, 0, 0, 1]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        2,
        np.asarray([[0, 0, 0, 0, 0, 1]], dtype=DTYPE_STATE),
        np.asarray([[1]], dtype=DTYPE_ACTION),
        np.asarray([[0, 0, 1, 0, 0, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.TERMINATED,
    ),
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        True,
        False,
        0,
        np.asarray([[0, 0, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[1]], dtype=DTYPE_ACTION),
        np.asarray([[1, 0, 0, 1]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        True,
        0,
        np.asarray([[0, 0, 0, 1, 0, 0]], dtype=DTYPE_STATE),
        np.asarray([[1]], dtype=DTYPE_ACTION),
        np.asarray([[1, 0, 0, 0, 1, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        True,
        True,
        0,
        np.asarray([[0, 0, 0, 0, 1, 0, 0, 0]], dtype=DTYPE_STATE),
        np.asarray([[1]], dtype=DTYPE_ACTION),
        np.asarray([[1, 0, 0, 0, 0, 1, 0, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
]

TEST_CASES_TRANSITION_BATCH_ROW_MAJOR += [
    *batchify(
        [test_case for test_case in TEST_CASES_TRANSITION_BATCH_ROW_MAJOR if test_case[0] == 1],
        batch_size=2,
        expand_dims=False,
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

TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH_ROW_MAJOR = [
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        np.asarray([[0, 0]], dtype=DTYPE_STATE),
        np.asarray([[0]], dtype=DTYPE_STATE),
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        np.asarray([[0, 0, 0, 0, 0, 0]], dtype=DTYPE_STATE),
        np.asarray([[0, 0, 0]], dtype=DTYPE_STATE),
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        np.asarray([[1, 0, 1, 0, 0, 0]], dtype=DTYPE_STATE),
        np.asarray([[1, 0, 1]], dtype=DTYPE_STATE),
    ),
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        np.asarray([[0, 1]], dtype=DTYPE_STATE),
        np.asarray([[2]], dtype=DTYPE_STATE),
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        np.asarray([[0, 0, 0, 0, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[0, 2, 2]], dtype=DTYPE_STATE),
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        np.asarray([[1, 0, 0, 0, 0, 1]], dtype=DTYPE_STATE),
        np.asarray([[1, 0, 2]], dtype=DTYPE_STATE),
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        3,
        False,
        False,
        np.asarray([[1, 0, 0, 0, 1, 0, 0, 0, 1]], dtype=DTYPE_STATE),
        np.asarray([[1, 2, 3]], dtype=DTYPE_STATE),
    ),
]

TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH_ROW_MAJOR += [
    *batchify(
        [
            test_case
            for test_case in TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH_ROW_MAJOR
            if test_case[0] == 1
        ],
        batch_size=2,
        expand_dims=False,
    ),
]

TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH_CLOCKWISE = replace(
    TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH_ROW_MAJOR,
    FlattenedOrdering.ROW_MAJOR,
    FlattenedOrdering.CLOCKWISE,
)

TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH = (
    TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH_ROW_MAJOR
    + TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH_CLOCKWISE
)
