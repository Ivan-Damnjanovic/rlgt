"""
This file is used for testing the functionalities from the `rl_graph_theory.graphs.graph` module.
"""

import numpy as np
import pytest

from rl_graph_theory.graphs.graph import Graph
from rl_graph_theory.graphs.graph_batch import GraphBatch
from rl_graph_theory.graphs.graph_format import BitmaskType, FlattenedOrdering

from .graph_test_cases import (
    GRAPH_BATCH_TEST_CASES_BASIC,
    GRAPH_BATCH_TEST_CASES_DIRECTED,
    GRAPH_BATCH_TEST_CASES_DIRECTED_LOOPS,
    GRAPH_BATCH_TEST_CASES_LARGE,
    GRAPH_BATCH_TEST_CASES_LOOPS,
    GRAPH_TEST_CASES_BASIC,
    GRAPH_TEST_CASES_DIRECTED,
    GRAPH_TEST_CASES_DIRECTED_LOOPS,
    GRAPH_TEST_CASES_LARGE,
    GRAPH_TEST_CASES_LOOPS,
)
from .utils import verify_instantiated_graph, verify_instantiated_graph_batch


@pytest.mark.parametrize(
    "edge_colors, order, bitmask, adjacency_matrix, flattened_clockwise, flattened_row_major",
    GRAPH_TEST_CASES_LARGE,
)
def test_graph_large(
    edge_colors: int,
    order: int,
    bitmask: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_clockwise: np.ndarray,
    flattened_row_major: np.ndarray,
):
    verify_all(
        edge_colors,
        order,
        bitmask,
        bitmask,
        adjacency_matrix,
        flattened_clockwise,
        flattened_row_major,
    )


@pytest.mark.parametrize(
    "edge_colors, order, bitmask, adjacency_matrix, flattened_clockwise, flattened_row_major",
    GRAPH_TEST_CASES_BASIC,
)
def test_graph_basic(
    edge_colors: int,
    order: int,
    bitmask: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_clockwise: np.ndarray,
    flattened_row_major: np.ndarray,
):
    verify_all(
        edge_colors,
        order,
        bitmask,
        bitmask,
        adjacency_matrix,
        flattened_clockwise,
        flattened_row_major,
    )


@pytest.mark.parametrize(
    "edge_colors, order, bitmask, adjacency_matrix, flattened_clockwise, flattened_row_major",
    GRAPH_TEST_CASES_LOOPS,
)
def test_graph_loops(
    edge_colors: int,
    order: int,
    bitmask: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_clockwise: np.ndarray,
    flattened_row_major: np.ndarray,
):
    verify_all(
        edge_colors,
        order,
        bitmask,
        bitmask,
        adjacency_matrix,
        flattened_clockwise,
        flattened_row_major,
        allow_loops=True,
    )


@pytest.mark.parametrize(
    "edge_colors, order, bitmask_in, bitmask_out, adjacency_matrix, flattened_clockwise, flattened_row_major",
    GRAPH_TEST_CASES_DIRECTED,
)
def test_graph_directed(
    edge_colors: int,
    order: int,
    bitmask_in: np.ndarray,
    bitmask_out: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_clockwise: np.ndarray,
    flattened_row_major: np.ndarray,
):
    verify_all(
        edge_colors,
        order,
        bitmask_in,
        bitmask_out,
        adjacency_matrix,
        flattened_clockwise,
        flattened_row_major,
        is_directed=True,
    )


@pytest.mark.parametrize(
    "edge_colors, order, bitmask_in, bitmask_out, adjacency_matrix, flattened_clockwise, flattened_row_major",
    GRAPH_TEST_CASES_DIRECTED_LOOPS,
)
def test_graph_directed_loops(
    edge_colors: int,
    order: int,
    bitmask_in: np.ndarray,
    bitmask_out: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_clockwise: np.ndarray,
    flattened_row_major: np.ndarray,
):
    verify_all(
        edge_colors,
        order,
        bitmask_in,
        bitmask_out,
        adjacency_matrix,
        flattened_clockwise,
        flattened_row_major,
        is_directed=True,
        allow_loops=True,
    )


def verify_all(
    edge_colors: int,
    order: int,
    bitmask_in: np.ndarray,
    bitmask_out: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_clockwise: np.ndarray,
    flattened_row_major: np.ndarray,
    is_directed: bool = False,
    allow_loops: bool = False,
):
    cargs = [edge_colors, is_directed, allow_loops]
    args = [
        edge_colors,
        order,
        bitmask_in,
        bitmask_out,
        adjacency_matrix,
        flattened_clockwise,
        flattened_row_major,
        is_directed,
        allow_loops,
    ]
    verify_instantiated_graph(
        lambda: Graph.from_bitmask(bitmask_out, BitmaskType.OUT_NEIGHBORS, *cargs),
        *args,
    )
    verify_instantiated_graph(
        lambda: Graph.from_bitmask(bitmask_in, BitmaskType.IN_NEIGHBORS, *cargs),
        *args,
    )
    verify_instantiated_graph(
        lambda: Graph.from_adjacency_matrix(adjacency_matrix, *cargs),
        *args,
    )
    verify_instantiated_graph(
        lambda: Graph.from_flattened(flattened_clockwise, FlattenedOrdering.CLOCKWISE, *cargs),
        *args,
    )
    verify_instantiated_graph(
        lambda: Graph.from_flattened(flattened_row_major, FlattenedOrdering.ROW_MAJOR, *cargs),
        *args,
    )


@pytest.mark.parametrize(
    "batch_size, edge_colors, order, bitmask, adjacency_matrix, "
    "flattened_clockwise, flattened_row_major",
    GRAPH_BATCH_TEST_CASES_BASIC,
)
def test_graph_batch_basic(
    batch_size: int,
    edge_colors: int,
    order: int,
    bitmask: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_clockwise: np.ndarray,
    flattened_row_major: np.ndarray,
):
    verify_all_batch(
        batch_size,
        edge_colors,
        order,
        bitmask,
        bitmask,
        adjacency_matrix,
        flattened_clockwise,
        flattened_row_major,
    )


@pytest.mark.parametrize(
    "batch_size, edge_colors, order, bitmask, adjacency_matrix, "
    "flattened_clockwise, flattened_row_major",
    GRAPH_BATCH_TEST_CASES_LOOPS,
)
def test_graph_batch_loops(
    batch_size: int,
    edge_colors: int,
    order: int,
    bitmask: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_clockwise: np.ndarray,
    flattened_row_major: np.ndarray,
):
    verify_all_batch(
        batch_size,
        edge_colors,
        order,
        bitmask,
        bitmask,
        adjacency_matrix,
        flattened_clockwise,
        flattened_row_major,
        allow_loops=True,
    )


@pytest.mark.parametrize(
    "batch_size, edge_colors, order, bitmask_in, bitmask_out, adjacency_matrix, "
    "flattened_clockwise, flattened_row_major",
    GRAPH_BATCH_TEST_CASES_DIRECTED,
)
def test_graph_batch_directed(
    batch_size: int,
    edge_colors: int,
    order: int,
    bitmask_in: np.ndarray,
    bitmask_out: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_clockwise: np.ndarray,
    flattened_row_major: np.ndarray,
):
    verify_all_batch(
        batch_size,
        edge_colors,
        order,
        bitmask_in,
        bitmask_out,
        adjacency_matrix,
        flattened_clockwise,
        flattened_row_major,
        is_directed=True,
    )


@pytest.mark.parametrize(
    "batch_size, edge_colors, order, bitmask_in, bitmask_out, adjacency_matrix, "
    "flattened_clockwise, flattened_row_major",
    GRAPH_BATCH_TEST_CASES_DIRECTED_LOOPS,
)
def test_graph_batch_directed_loops(
    batch_size: int,
    edge_colors: int,
    order: int,
    bitmask_in: np.ndarray,
    bitmask_out: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_clockwise: np.ndarray,
    flattened_row_major: np.ndarray,
):
    verify_all_batch(
        batch_size,
        edge_colors,
        order,
        bitmask_in,
        bitmask_out,
        adjacency_matrix,
        flattened_clockwise,
        flattened_row_major,
        is_directed=True,
        allow_loops=True,
    )


@pytest.mark.parametrize(
    "batch_size, edge_colors, order, bitmask, adjacency_matrix, "
    "flattened_clockwise, flattened_row_major",
    GRAPH_BATCH_TEST_CASES_LARGE,
)
def test_graph_batch_large(
    batch_size: int,
    edge_colors: int,
    order: int,
    bitmask: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_clockwise: np.ndarray,
    flattened_row_major: np.ndarray,
):
    verify_all_batch(
        batch_size,
        edge_colors,
        order,
        bitmask,
        bitmask,
        adjacency_matrix,
        flattened_clockwise,
        flattened_row_major,
    )


def verify_all_batch(
    batch_size: int,
    edge_colors: int,
    order: int,
    bitmask_in: np.ndarray,
    bitmask_out: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_clockwise: np.ndarray,
    flattened_row_major: np.ndarray,
    is_directed: bool = False,
    allow_loops: bool = False,
):
    cargs = [edge_colors, is_directed, allow_loops]
    args = [
        batch_size,
        edge_colors,
        order,
        bitmask_in,
        bitmask_out,
        adjacency_matrix,
        flattened_clockwise,
        flattened_row_major,
        is_directed,
        allow_loops,
    ]
    verify_instantiated_graph_batch(
        lambda: GraphBatch.from_bitmask(bitmask_out, BitmaskType.OUT_NEIGHBORS, *cargs),
        *args,
    )
    verify_instantiated_graph_batch(
        lambda: GraphBatch.from_bitmask(bitmask_in, BitmaskType.IN_NEIGHBORS, *cargs),
        *args,
    )
    verify_instantiated_graph_batch(
        lambda: GraphBatch.from_adjacency_matrix(adjacency_matrix, *cargs),
        *args,
    )
    verify_instantiated_graph_batch(
        lambda: GraphBatch.from_flattened(
            flattened_clockwise, FlattenedOrdering.CLOCKWISE, *cargs
        ),
        *args,
    )
    verify_instantiated_graph_batch(
        lambda: GraphBatch.from_flattened(
            flattened_row_major, FlattenedOrdering.ROW_MAJOR, *cargs
        ),
        *args,
    )
