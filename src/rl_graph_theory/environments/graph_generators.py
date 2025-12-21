"""
#TODO
"""

from typing import Callable

import numpy as np

from ..graphs.graph import Graph
from ..graphs.graph_batch import GraphBatch
from ..graphs.graph_format import GraphFormat


GraphGenerator = Callable[[int], GraphBatch]


def create_fixed_graph_generator(
    fixed_graph: Graph, graph_format: GraphFormat = GraphFormat.FLATTENED_ROW_MAJOR
) -> GraphGenerator:
    """
    #TODO
    """

    if graph_format == GraphFormat.FLATTENED_ROW_MAJOR:
        input_representation = fixed_graph.flattened_row_major
    elif graph_format == GraphFormat.FLATTENED_CLOCKWISE:
        input_representation = fixed_graph.flattened_clockwise
    elif graph_format == GraphFormat.BITMASK_OUT:
        input_representation = fixed_graph.bitmask_out
    elif graph_format == GraphFormat.BITMASK_IN:
        input_representation = fixed_graph.bitmask_in
    else:
        input_representation = fixed_graph.adjacency_matrix

    def result(batch_size: int) -> GraphBatch:
        format_representation = np.empty(
            (batch_size, *input_representation.shape), dtype=input_representation.dtype
        )
        format_representation[:] = input_representation

        return GraphBatch(
            graph_format=graph_format,
            bitmask_out=format_representation,
            bitmask_in=format_representation,
            adjacency_matrix=format_representation,
            flattened_row_major=format_representation,
            flattened_clockwise=format_representation,
            edge_colors=fixed_graph.edge_colors,
            is_directed=fixed_graph.is_directed,
            allow_loops=fixed_graph.allow_loops,
        )

    return result
