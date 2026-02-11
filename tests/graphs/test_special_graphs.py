"""
This file is used for testing the functionalities from the
`rlgt.graphs.special_graphs` module.
"""

from typing import List

import numpy as np
import pytest

from rlgt.graphs.graph import GraphFormat
from rlgt.graphs.special_graphs import (
    AlmostCompleteGraph,
    BookGraph,
    CompleteBipartiteGraph,
    CompleteGraph,
    CompleteKPartiteGraph,
    CycleGraph,
    EmptyGraph,
    FriendshipGraph,
    MonochromaticGraph,
    PathGraph,
    StarGraph,
    WheelGraph,
)

from .special_graphs_test_cases import (
    ALMOST_COMPLETE_GRAPH_TEST_CASES,
    BOOK_GRAPH_TEST_CASES,
    COMPLETE_BIPARTITE_GRAPH_TEST_CASES,
    COMPLETE_GRAPH_TEST_CASES,
    COMPLETE_K_PARTITE_GRAPH_TEST_CASES,
    CYCLE_GRAPH_TEST_CASES,
    EMPTY_GRAPH_TEST_CASES,
    FRIENDSHIP_GRAPH_TEST_CASES,
    MONOCHROMATIC_GRAPH_TEST_CASES,
    PATH_GRAPH_TEST_CASES,
    STAR_GRAPH_TEST_CASES,
    WHEEL_GRAPH_TEST_CASES,
)
from .utils import verify_instantiated_graph

ALL_FORMATS = {format for format in GraphFormat}


@pytest.mark.parametrize(
    "order, edge_colors, selected_edge_color, bitmask_in, bitmask_out, adjacency_matrix, flattened, is_directed, allow_loops",
    MONOCHROMATIC_GRAPH_TEST_CASES,
)
def test_monochromatic_graph(
    order: int,
    edge_colors: int,
    selected_edge_color: int,
    bitmask_in: np.ndarray,
    bitmask_out: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened: np.ndarray,
    is_directed: bool,
    allow_loops: bool,
):
    verify_instantiated_graph(
        constructor=lambda: MonochromaticGraph(
            graph_formats=ALL_FORMATS,
            graph_order=order,
            edge_colors=edge_colors,
            selected_color=selected_edge_color,
            is_directed=is_directed,
            allow_loops=allow_loops,
        ),
        edge_colors=edge_colors,
        order=order,
        bitmask_in=bitmask_in,
        bitmask_out=bitmask_out,
        adjacency_matrix=adjacency_matrix,
        flattened_clockwise=flattened,
        flattened_row_major=flattened,
        is_directed=is_directed,
        allow_loops=allow_loops,
    )


@pytest.mark.parametrize("order, bitmask, adjacency_matrix, flattened", EMPTY_GRAPH_TEST_CASES)
def test_empty_graph(
    order: int,
    bitmask: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened: np.ndarray,
):
    verify_instantiated_graph(
        constructor=lambda: EmptyGraph(ALL_FORMATS, order),
        edge_colors=2,
        order=order,
        bitmask_in=bitmask,
        bitmask_out=bitmask,
        adjacency_matrix=adjacency_matrix,
        flattened_clockwise=flattened,
        flattened_row_major=flattened,
    )


@pytest.mark.parametrize("order, bitmask, adjacency_matrix, flattened", COMPLETE_GRAPH_TEST_CASES)
def test_complete_graph(
    order: int,
    bitmask: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened: np.ndarray,
):
    verify_instantiated_graph(
        constructor=lambda: CompleteGraph(ALL_FORMATS, order),
        edge_colors=2,
        order=order,
        bitmask_in=bitmask,
        bitmask_out=bitmask,
        adjacency_matrix=adjacency_matrix,
        flattened_clockwise=flattened,
        flattened_row_major=flattened,
    )


@pytest.mark.parametrize(
    "order, bitmask, adjacency_matrix, flattened", ALMOST_COMPLETE_GRAPH_TEST_CASES
)
def test_almost_complete_graph(
    order: int,
    bitmask: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened: np.ndarray,
):
    verify_instantiated_graph(
        constructor=lambda: AlmostCompleteGraph(ALL_FORMATS, order),
        edge_colors=2,
        order=order,
        bitmask_in=bitmask,
        bitmask_out=bitmask,
        adjacency_matrix=adjacency_matrix,
        flattened_clockwise=flattened,
        flattened_row_major=flattened,
    )


@pytest.mark.parametrize(
    "partition_size_1, partition_size_2, order, bitmask, adjacency_matrix, "
    "flattened_column_first, flattened_row_first",
    COMPLETE_BIPARTITE_GRAPH_TEST_CASES,
)
def test_complete_bipartite_graph(
    partition_size_1: int,
    partition_size_2: int,
    order: int,
    bitmask: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_column_first: np.ndarray,
    flattened_row_first: np.ndarray,
):
    verify_instantiated_graph(
        constructor=lambda: CompleteBipartiteGraph(
            ALL_FORMATS, partition_size_1, partition_size_2
        ),
        edge_colors=2,
        order=order,
        bitmask_in=bitmask,
        bitmask_out=bitmask,
        adjacency_matrix=adjacency_matrix,
        flattened_clockwise=flattened_column_first,
        flattened_row_major=flattened_row_first,
    )


@pytest.mark.parametrize(
    "partition_sizes, order, bitmask, adjacency_matrix, flattened_column_first, "
    "flattened_row_first",
    COMPLETE_K_PARTITE_GRAPH_TEST_CASES,
)
def test_complete_k_partite_graph(
    partition_sizes: List[int],
    order: int,
    bitmask: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_column_first: np.ndarray,
    flattened_row_first: np.ndarray,
):
    verify_instantiated_graph(
        constructor=lambda: CompleteKPartiteGraph(ALL_FORMATS, partition_sizes),
        edge_colors=2,
        order=order,
        bitmask_in=bitmask,
        bitmask_out=bitmask,
        adjacency_matrix=adjacency_matrix,
        flattened_clockwise=flattened_column_first,
        flattened_row_major=flattened_row_first,
    )


@pytest.mark.parametrize(
    "order, central_vertex, bitmask, adjacency_matrix, flattened_column_first, "
    "flattened_row_first",
    STAR_GRAPH_TEST_CASES,
)
def test_star_graph(
    order: int,
    central_vertex: int,
    bitmask: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_column_first: np.ndarray,
    flattened_row_first: np.ndarray,
):
    verify_instantiated_graph(
        constructor=lambda: StarGraph(ALL_FORMATS, order, central_vertex),
        edge_colors=2,
        order=order,
        bitmask_in=bitmask,
        bitmask_out=bitmask,
        adjacency_matrix=adjacency_matrix,
        flattened_clockwise=flattened_column_first,
        flattened_row_major=flattened_row_first,
    )


@pytest.mark.parametrize(
    "order, bitmask, adjacency_matrix, flattened_column_first, flattened_row_first",
    PATH_GRAPH_TEST_CASES,
)
def test_path_graph(
    order: int,
    bitmask: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_column_first: np.ndarray,
    flattened_row_first: np.ndarray,
):
    verify_instantiated_graph(
        constructor=lambda: PathGraph(ALL_FORMATS, order),
        edge_colors=2,
        order=order,
        bitmask_in=bitmask,
        bitmask_out=bitmask,
        adjacency_matrix=adjacency_matrix,
        flattened_clockwise=flattened_column_first,
        flattened_row_major=flattened_row_first,
    )


@pytest.mark.parametrize(
    "order, bitmask, adjacency_matrix, flattened_column_first, flattened_row_first",
    CYCLE_GRAPH_TEST_CASES,
)
def test_cycle_graph(
    order: int,
    bitmask: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_column_first: np.ndarray,
    flattened_row_first: np.ndarray,
):
    verify_instantiated_graph(
        constructor=lambda: CycleGraph(ALL_FORMATS, order),
        edge_colors=2,
        order=order,
        bitmask_out=bitmask,
        bitmask_in=bitmask,
        adjacency_matrix=adjacency_matrix,
        flattened_clockwise=flattened_column_first,
        flattened_row_major=flattened_row_first,
    )


@pytest.mark.parametrize(
    "order, bitmask, adjacency_matrix, flattened_column_first, flattened_row_first",
    WHEEL_GRAPH_TEST_CASES,
)
def test_wheel_graph(
    order: int,
    bitmask: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_column_first: np.ndarray,
    flattened_row_first: np.ndarray,
):
    verify_instantiated_graph(
        constructor=lambda: WheelGraph(ALL_FORMATS, order),
        edge_colors=2,
        order=order,
        bitmask_out=bitmask,
        bitmask_in=bitmask,
        adjacency_matrix=adjacency_matrix,
        flattened_clockwise=flattened_column_first,
        flattened_row_major=flattened_row_first,
    )


@pytest.mark.parametrize(
    "index, bitmask, adjacency_matrix, flattened_column_first, flattened_row_first",
    BOOK_GRAPH_TEST_CASES,
)
def test_book_graph(
    index: int,
    bitmask: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_column_first: np.ndarray,
    flattened_row_first: np.ndarray,
):
    verify_instantiated_graph(
        constructor=lambda: BookGraph(ALL_FORMATS, index),
        edge_colors=2,
        order=index + 2,
        bitmask_out=bitmask,
        bitmask_in=bitmask,
        adjacency_matrix=adjacency_matrix,
        flattened_clockwise=flattened_column_first,
        flattened_row_major=flattened_row_first,
    )


@pytest.mark.parametrize(
    "index, bitmask, adjacency_matrix, flattened_column_first, flattened_row_first",
    FRIENDSHIP_GRAPH_TEST_CASES,
)
def test_friendship_graph(
    index: int,
    bitmask: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_column_first: np.ndarray,
    flattened_row_first: np.ndarray,
):
    verify_instantiated_graph(
        constructor=lambda: FriendshipGraph(ALL_FORMATS, index),
        edge_colors=2,
        order=2 * index + 1,
        bitmask_in=bitmask,
        bitmask_out=bitmask,
        adjacency_matrix=adjacency_matrix,
        flattened_clockwise=flattened_column_first,
        flattened_row_major=flattened_row_first,
    )
