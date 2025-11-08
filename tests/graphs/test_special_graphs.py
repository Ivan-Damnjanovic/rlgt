"""
This file is used for testing the functionalities from the `graphs.special_graphs` module.
"""

import numpy as np
import pytest

from .special_graphs_test_cases import COMPLETE_BIPARTITE_GRAPH_TEST_CASES
from .utils import verify_instantiated_graph

from rl_graph_theory.graphs.graph import GraphFormat
from rl_graph_theory.graphs.special_graphs import CompleteBipartiteGraph


@pytest.mark.parametrize(
    "partition_size_1, partition_size_2, bitmask, adjacency_matrix, flattened_column_first, "
    "flattened_row_first",
    COMPLETE_BIPARTITE_GRAPH_TEST_CASES,
)
def test_complete_bipartite_graph(
    partition_size_1: int,
    partition_size_2: int,
    bitmask: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_column_first: np.ndarray,
    flattened_row_first: np.ndarray,
):
    """
    #TODO
    """

    verify_instantiated_graph(
        constructor=lambda: CompleteBipartiteGraph(
            GraphFormat.BITMASK, partition_size_1, partition_size_2
        ),
        edge_colors=2,
        order=partition_size_1 + partition_size_2,
        bitmask=bitmask,
        adjacency_matrix=adjacency_matrix,
        flattened_column_first=flattened_column_first,
        flattened_row_first=flattened_row_first,
    )
    verify_instantiated_graph(
        constructor=lambda: CompleteBipartiteGraph(
            GraphFormat.ADJACENCY_MATRIX, partition_size_1, partition_size_2
        ),
        edge_colors=2,
        order=partition_size_1 + partition_size_2,
        bitmask=bitmask,
        adjacency_matrix=adjacency_matrix,
        flattened_column_first=flattened_column_first,
        flattened_row_first=flattened_row_first,
    )
