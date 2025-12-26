from copy import deepcopy

from rl_graph_theory.graphs.graph import FlattenedOrdering

from ..utils import batchify, replace
from .global_set_test_cases import TEST_CASES_CONSTRUCTOR as gstc_constructor

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

TEST_CASES_RESET_BATCH = (
    TEST_CASES_RESET_BATCH_ROW_MAJOR + TEST_CASES_RESET_BATCH_CLOCKWISE
)
