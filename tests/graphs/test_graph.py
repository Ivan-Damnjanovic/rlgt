"""
This file is used for testing the functionalities from the `rl_graph_theory.graphs.graph` module.
"""

import numpy as np
import pytest

from rl_graph_theory.graphs.graph import FlattenedOrdering, Graph, GraphBatch

from .graph_test_cases import GRAPH_BATCH_TEST_CASES, GRAPH_TEST_CASES
from .utils import verify_instantiated_graph, verify_instantiated_graph_batch


@pytest.mark.parametrize(
    "edge_colors, order, bitmask, adjacency_matrix, flattened_column_first, flattened_row_first",
    GRAPH_TEST_CASES,
)
def test_graph(
    edge_colors: int,
    order: int,
    bitmask: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_column_first: np.ndarray,
    flattened_row_first: np.ndarray,
):
    verify_instantiated_graph(
        constructor=lambda: Graph.from_bitmask(edge_colors, bitmask),
        edge_colors=edge_colors,
        order=order,
        bitmask=bitmask,
        adjacency_matrix=adjacency_matrix,
        flattened_column_first=flattened_column_first,
        flattened_row_first=flattened_row_first,
    )
    verify_instantiated_graph(
        constructor=lambda: Graph.from_adjacency_matrix(edge_colors, adjacency_matrix),
        edge_colors=edge_colors,
        order=order,
        bitmask=bitmask,
        adjacency_matrix=adjacency_matrix,
        flattened_column_first=flattened_column_first,
        flattened_row_first=flattened_row_first,
    )
    verify_instantiated_graph(
        constructor=lambda: Graph.from_flattened(
            edge_colors, FlattenedOrdering.CLOCKWISE, flattened_column_first
        ),
        edge_colors=edge_colors,
        order=order,
        bitmask=bitmask,
        adjacency_matrix=adjacency_matrix,
        flattened_column_first=flattened_column_first,
        flattened_row_first=flattened_row_first,
    )
    verify_instantiated_graph(
        constructor=lambda: Graph.from_flattened(
            edge_colors, FlattenedOrdering.ROW_MAJOR, flattened_row_first
        ),
        edge_colors=edge_colors,
        order=order,
        bitmask=bitmask,
        adjacency_matrix=adjacency_matrix,
        flattened_column_first=flattened_column_first,
        flattened_row_first=flattened_row_first,
    )


@pytest.mark.parametrize(
    "batch_size, edge_colors, order, bitmask_batch, adjacency_matrix_batch, "
    "flattened_column_first_batch, flattened_row_first_batch",
    GRAPH_BATCH_TEST_CASES,
)
def test_graph_batch(
    batch_size: int,
    edge_colors: int,
    order: int,
    bitmask_batch: np.ndarray,
    adjacency_matrix_batch: np.ndarray,
    flattened_column_first_batch: np.ndarray,
    flattened_row_first_batch: np.ndarray,
):
    verify_instantiated_graph_batch(
        constructor=lambda: GraphBatch.from_bitmask_batch(edge_colors, bitmask_batch),
        batch_size=batch_size,
        edge_colors=edge_colors,
        order=order,
        bitmask_batch=bitmask_batch,
        adjacency_matrix_batch=adjacency_matrix_batch,
        flattened_column_first_batch=flattened_column_first_batch,
        flattened_row_first_batch=flattened_row_first_batch,
    )
    verify_instantiated_graph_batch(
        constructor=lambda: GraphBatch.from_adjacency_matrix_batch(
            edge_colors, adjacency_matrix_batch
        ),
        batch_size=batch_size,
        edge_colors=edge_colors,
        order=order,
        bitmask_batch=bitmask_batch,
        adjacency_matrix_batch=adjacency_matrix_batch,
        flattened_column_first_batch=flattened_column_first_batch,
        flattened_row_first_batch=flattened_row_first_batch,
    )
    verify_instantiated_graph_batch(
        constructor=lambda: GraphBatch.from_flattened_batch(
            edge_colors,
            FlattenedOrdering.CLOCKWISE,
            flattened_column_first_batch,
        ),
        batch_size=batch_size,
        edge_colors=edge_colors,
        order=order,
        bitmask_batch=bitmask_batch,
        adjacency_matrix_batch=adjacency_matrix_batch,
        flattened_column_first_batch=flattened_column_first_batch,
        flattened_row_first_batch=flattened_row_first_batch,
    )
    verify_instantiated_graph_batch(
        constructor=lambda: GraphBatch.from_flattened_batch(
            edge_colors,
            FlattenedOrdering.ROW_MAJOR,
            flattened_row_first_batch,
        ),
        batch_size=batch_size,
        edge_colors=edge_colors,
        order=order,
        bitmask_batch=bitmask_batch,
        adjacency_matrix_batch=adjacency_matrix_batch,
        flattened_column_first_batch=flattened_column_first_batch,
        flattened_row_first_batch=flattened_row_first_batch,
    )
