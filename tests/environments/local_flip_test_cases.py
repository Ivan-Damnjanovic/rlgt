import numpy as np

from rlgt.environments.graph_environment import EpisodeStatus
from rlgt.graphs.graph import FlattenedOrdering

from ..utils import batchify, insert, remove_at, replace
from .local_set_test_cases import TEST_CASES_CONSTRUCTOR as lstc_constructor
from .local_set_test_cases import TEST_CASES_RESET_BATCH as lstc_reset_batch
from .local_set_test_cases import (
    TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH as lstc_state_batch_to_graph_batch,
)


DTYPE_STATE = np.uint8
DTYPE_ACTION = np.int32

TEST_CASES_CONSTRUCTOR = insert(
    remove_at([tc for tc in lstc_constructor if tc[5] == 2], 5), 4, False
) + insert(remove_at([tc for tc in lstc_constructor if tc[5] == 2], 5), 4, True)

TEST_CASES_RESET_BATCH = remove_at([tc for tc in lstc_reset_batch if tc[3] == 2], 3)

TEST_CASES_TRANSITION_BATCH_ROW_MAJOR = [
    (
        1,
        2,
        2,
        True,
        FlattenedOrdering.ROW_MAJOR,
        False,
        False,
        0,
        np.asarray([[1, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[1]], dtype=DTYPE_ACTION),
        np.asarray([[0, 0, 1]], dtype=DTYPE_STATE),
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
        np.asarray([[0, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[1]], dtype=DTYPE_ACTION),
        np.asarray([[1, 0, 1]], dtype=DTYPE_STATE),
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
        np.asarray([[0, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[1, 1]], dtype=DTYPE_ACTION),
        np.asarray([[1, 0, 1]], dtype=DTYPE_STATE),
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
        1,
        np.asarray([[0, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[1, 1]], dtype=DTYPE_ACTION),
        np.asarray([[1, 0, 1]], dtype=DTYPE_STATE),
        EpisodeStatus.TRUNCATED,
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
        np.asarray([[1, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[1, 1]], dtype=DTYPE_ACTION),
        np.asarray([[0, 0, 1]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        2,
        False,
        FlattenedOrdering.ROW_MAJOR,
        True,
        False,
        0,
        np.asarray([[1, 0, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[1, 1]], dtype=DTYPE_ACTION),
        np.asarray([[0, 0, 0, 1]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        2,
        False,
        FlattenedOrdering.ROW_MAJOR,
        False,
        True,
        0,
        np.asarray([[1, 0, 0, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[1, 1]], dtype=DTYPE_ACTION),
        np.asarray([[1, 1, 0, 0, 1]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        2,
        False,
        FlattenedOrdering.ROW_MAJOR,
        True,
        True,
        0,
        np.asarray([[1, 0, 0, 0, 1, 0]], dtype=DTYPE_STATE),
        np.asarray([[1, 1]], dtype=DTYPE_ACTION),
        np.asarray([[1, 1, 0, 0, 0, 1]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        3,
        2,
        False,
        FlattenedOrdering.ROW_MAJOR,
        False,
        False,
        0,
        np.asarray([[1, 0, 0, 1, 0, 0]], dtype=DTYPE_STATE),
        np.asarray([[1, 1]], dtype=DTYPE_ACTION),
        np.asarray([[0, 0, 0, 0, 1, 0]], dtype=DTYPE_STATE),
        EpisodeStatus.IN_PROGRESS,
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


TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH = remove_at(
    [tc for tc in lstc_state_batch_to_graph_batch if tc[3] == 2], 3
)
