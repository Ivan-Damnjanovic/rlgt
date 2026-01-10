from copy import deepcopy

import numpy as np

from rl_graph_theory.environments.graph_environment import EpisodeStatus
from rl_graph_theory.graphs.graph import FlattenedOrdering

from ..utils import batchify, replace
from .global_set_test_cases import TEST_CASES_CONSTRUCTOR as gstc_constructor

DTYPE_STATE = np.uint8
DTYPE_ACTION = np.int32

TEST_CASES_CONSTRUCTOR = deepcopy(gstc_constructor)


TEST_CASES_RESET_BATCH_ROW_MAJOR = [
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        [[0, 1, 0]],
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
        [[0, 0, 0, 1, 0]],
    ),
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        True,
        [[0, 0, 0, 1, 0]],
    ),
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        True,
        True,
        [[0, 0, 0, 0, 1, 0]],
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
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        True,
        False,
        [[0, 0, 0, 0, 0, 0, 1, 0, 0]],
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        True,
        [[0, 0, 0, 0, 0, 0, 1, 0, 0]],
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        True,
        True,
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        3,
        False,
        False,
        [[0, 0, 0, 0, 0, 0, 1, 0, 0]],
    ),
]

TEST_CASES_RESET_BATCH_ROW_MAJOR += [
    *batchify(
        TEST_CASES_RESET_BATCH_ROW_MAJOR,
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
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        0,
        np.asarray([[0, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[1, 1]], dtype=DTYPE_ACTION),
        np.asarray([[1, 0, 1]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        True,
        False,
        0,
        np.asarray([[0, 0, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[1, 1]], dtype=DTYPE_ACTION),
        np.asarray([[1, 0, 0, 1]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        True,
        False,
        1,
        np.asarray([[0, 0, 0, 1]], dtype=DTYPE_STATE),
        np.asarray([[0, 1]], dtype=DTYPE_ACTION),
        np.asarray([[0, 1, 1, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.TRUNCATED,
    ),
    (
        1,
        2,
        2,
        FlattenedOrdering.ROW_MAJOR,
        3,
        True,
        False,
        1,
        np.asarray([[0, 0, 0, 0, 0, 1]], dtype=DTYPE_STATE),
        np.asarray([[0, 2]], dtype=DTYPE_ACTION),
        np.asarray([[0, 0, 0, 1, 1, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.TRUNCATED,
    ),
    (
        1,
        2,
        2,
        FlattenedOrdering.ROW_MAJOR,
        3,
        True,
        False,
        1,
        np.asarray([[0, 1, 0, 0, 0, 1]], dtype=DTYPE_STATE),
        np.asarray([[0, 2]], dtype=DTYPE_ACTION),
        np.asarray([[0, 0, 0, 1, 1, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.TRUNCATED,
    ),
    (
        1,
        2,
        2,
        FlattenedOrdering.ROW_MAJOR,
        3,
        True,
        False,
        1,
        np.asarray([[1, 0, 0, 1, 0, 1]], dtype=DTYPE_STATE),
        np.asarray([[0, 0]], dtype=DTYPE_ACTION),
        np.asarray([[1, 0, 0, 0, 1, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.TRUNCATED,
    ),
    (
        1,
        2,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        True,
        1,
        np.asarray([[0, 0, 0, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[0, 1]], dtype=DTYPE_ACTION),
        np.asarray([[1, 0, 0, 1, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.TRUNCATED,
    ),
    (
        1,
        2,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        True,
        True,
        1,
        np.asarray([[0, 0, 0, 0, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[0, 1]], dtype=DTYPE_ACTION),
        np.asarray([[1, 0, 0, 0, 1, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.TRUNCATED,
    ),
]

TEST_CASES_TRANSITION_BATCH_ROW_MAJOR += [
    *batchify(
        TEST_CASES_TRANSITION_BATCH_ROW_MAJOR,
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
        np.asarray([[0, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[0]], dtype=DTYPE_STATE),
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        np.asarray([[0, 0, 0, 1, 0, 0]], dtype=DTYPE_STATE),
        np.asarray([[0, 0, 0]], dtype=DTYPE_STATE),
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        np.asarray([[1, 0, 1, 0, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[1, 0, 1]], dtype=DTYPE_STATE),
    ),
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        np.asarray([[0, 0, 1]], dtype=DTYPE_STATE),
        np.asarray([[0]], dtype=DTYPE_STATE),
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        np.asarray([[0, 0, 0, 0, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[0, 0, 0]], dtype=DTYPE_STATE),
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        np.asarray([[1, 0, 1, 1, 0, 0]], dtype=DTYPE_STATE),
        np.asarray([[1, 0, 1]], dtype=DTYPE_STATE),
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        3,
        False,
        False,
        np.asarray([[1, 0, 1, 0, 1, 0, 0, 0, 1]], dtype=DTYPE_STATE),
        np.asarray([[1, 2, 1]], dtype=DTYPE_STATE),
    ),
]

TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH_ROW_MAJOR += [
    *batchify(
        TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH_ROW_MAJOR,
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
