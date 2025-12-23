import numpy as np

from rl_graph_theory.graphs.graph import FlattenedOrdering
from rl_graph_theory.environments.graph_environment import EpisodeStatus

from ..utils import batchify, replace
from .linear_build_test_cases import TEST_CASES_CONSTRUCTOR as lbtc_constructor
from .linear_build_test_cases import TEST_CASES_RESET_BATCH as lbtc_reset_batch
from .linear_build_test_cases import TEST_CASES_TRANSITION_BATCH as lbtc_transition_batch

TEST_CASES_CONSTRUCTOR = [
    (*test_case[:-1], lambda *_: None, test_case[-1]) for test_case in lbtc_constructor
] + [(*test_case[:-1], None, test_case[-1]) for test_case in lbtc_constructor]

TEST_CASES_RESET_BATCH = [test_case + (None,) for test_case in lbtc_reset_batch]


TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH_ROW_MAJOR = [
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        np.asarray([[0, 0]], dtype=int),
        np.asarray([[0]], dtype=int),
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        np.asarray([[0, 0, 0, 0, 0, 0]], dtype=int),
        np.asarray([[0, 0, 0]], dtype=int),
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        np.asarray([[1, 0, 1, 0, 0, 0]], dtype=int),
        np.asarray([[1, 0, 1]], dtype=int),
    ),
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        np.asarray([[0, 1]], dtype=int),
        np.asarray([[0]], dtype=int),
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        np.asarray([[0, 0, 0, 0, 1, 0]], dtype=int),
        np.asarray([[0, 0, 0]], dtype=int),
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        False,
        np.asarray([[1, 0, 1, 0, 0, 1]], dtype=int),
        np.asarray([[1, 0, 1]], dtype=int),
    ),
    (
        1,
        3,
        FlattenedOrdering.ROW_MAJOR,
        3,
        False,
        False,
        np.asarray([[1, 0, 1, 0, 1, 0, 0, 0, 1]], dtype=int),
        np.asarray([[1, 2, 1]], dtype=int),
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
    + TEST_CASES_STATE_BATCH_TO_GRAPH_BATCH_ROW_MAJOR
)

TEST_CASES_TRANSITION_BATCH_ROW_MAJOR = [
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        2,
        False,
        True,
        0,
        np.asarray([[1, 0, 0, 1, 0, 0]], dtype=int),
        np.asarray([[0]], dtype=int),
        np.asarray([[0, 0, 0, 0, 1, 0]], dtype=int),
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
        np.asarray([[0, 0, 0, 1, 0, 0]], dtype=int),
        np.asarray([[1]], dtype=int),
        np.asarray([[1, 0, 0, 0, 1, 0]], dtype=int),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        3,
        False,
        True,
        0,
        np.asarray([[0, 0, 0, 0, 0, 0, 1, 0, 0]], dtype=int),
        np.asarray([[1]], dtype=int),
        np.asarray([[1, 0, 0, 0, 0, 0, 0, 1, 0]], dtype=int),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        3,
        False,
        True,
        0,
        np.asarray([[1, 0, 0, 0, 0, 0, 1, 0, 0]], dtype=int),
        np.asarray([[0]], dtype=int),
        np.asarray([[0, 0, 0, 0, 0, 0, 0, 1, 0]], dtype=int),
        EpisodeStatus.IN_PROGRESS,
    ),
    (
        1,
        2,
        FlattenedOrdering.ROW_MAJOR,
        3,
        False,
        True,
        0,
        np.asarray([[1, 0, 0, 0, 0, 0, 1, 0, 0]], dtype=int),
        np.asarray([[2]], dtype=int),
        np.asarray([[0, 0, 0, 1, 0, 0, 0, 1, 0]], dtype=int),
        EpisodeStatus.IN_PROGRESS,
    ),
]

TEST_CASES_TRANSITION_BATCH_ROW_MAJOR += [
    *batchify(
        [
            test_case
            for test_case in TEST_CASES_TRANSITION_BATCH_ROW_MAJOR[:2]
            if test_case[0] == 1
        ],
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
    lbtc_transition_batch
    + TEST_CASES_TRANSITION_BATCH_ROW_MAJOR
    + TEST_CASES_TRANSITION_BATCH_ROW_MAJOR
)
