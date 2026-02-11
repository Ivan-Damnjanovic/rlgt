import numpy as np

from rlgt.environments.graph_environment import EpisodeStatus
from rlgt.graphs.graph import FlattenedOrdering

from .linear_set_test_cases import TEST_CASES_CONSTRUCTOR as lstc_constructor
from .linear_set_test_cases import TEST_CASES_RESET_BATCH as lstc_reset_batch
from .linear_set_test_cases import (
    TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH as lstc_state_batch_to_graph_batch,
)
from ..utils import batchify, replace

DTYPE_STATE = np.uint8
DTYPE_ACTION = np.int32

TEST_CASES_CONSTRUCTOR = [
    test_case[:4] + test_case[5:] for test_case in lstc_constructor if test_case[4] == 2
]

TEST_CASES_RESET_BATCH = [
    test_case[:3] + test_case[4:] for test_case in lstc_reset_batch if test_case[3] == 2
]

TEST_CASES_TRANSITION_BATCH_ROW_MAJOR = [
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
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
        False,
        False,
        0,
        np.asarray([[1, 1]], dtype=DTYPE_STATE),
        np.asarray([[1]], dtype=DTYPE_ACTION),
        np.asarray([[0, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.TERMINATED,
    ),
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        False,
        False,
        0,
        np.asarray([[1, 1]], dtype=DTYPE_STATE),
        np.asarray([[0]], dtype=DTYPE_ACTION),
        np.asarray([[1, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.TERMINATED,
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        False,
        False,
        0,
        np.asarray([[1, 0, 0, 1, 0, 0]], dtype=DTYPE_STATE),
        np.asarray([[0]], dtype=DTYPE_ACTION),
        np.asarray([[1, 0, 0, 0, 1, 0]], dtype=DTYPE_STATE),
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
    TEST_CASES_TRANSITION_BATCH_ROW_MAJOR, FlattenedOrdering.ROW_MAJOR, FlattenedOrdering.CLOCKWISE
)

TEST_CASES_TRANSITION_BATCH = (
    TEST_CASES_TRANSITION_BATCH_ROW_MAJOR + TEST_CASES_TRANSITION_BATCH_CLOCKWISE
)

TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH = [
    test_case[:3] + test_case[4:]
    for test_case in lstc_state_batch_to_graph_batch
    if test_case[3] == 2
]
