"""
#TODO
"""

from typing import Callable, Optional

import numpy as np

from ..graphs.graph import Graph
from ..graphs.graph_batch import GraphBatch
from ..graphs.graph_format import GraphFormat, FlattenedOrdering


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


def create_choose_two_graph_generator(
    first_graph: Graph,
    second_graph: Graph,
    second_graph_probability: float,
    graph_format = GraphFormat.FLATTENED_ROW_MAJOR,
    rng: Optional[np.random.Generator] = None,
):
    """
    Docstring for create_choose_two_graph_generator
    
    :param first_graph: Description
    :type first_graph: Graph
    :param second_graph: Description
    :type second_graph: Graph
    :param second_graph_probability: Description
    :type second_graph_probability: float
    :param graph_format: Description
    """

    if rng is None:
        rng = np.random.default_rng()

    if graph_format == GraphFormat.FLATTENED_ROW_MAJOR:
        input_representation_1 = first_graph.flattened_row_major
        input_representation_2 = second_graph.flattened_row_major
    elif graph_format == GraphFormat.FLATTENED_CLOCKWISE:
        input_representation_1 = first_graph.flattened_clockwise
        input_representation_2 = second_graph.flattened_clockwise
    elif graph_format == GraphFormat.BITMASK_OUT:
        input_representation_1 = first_graph.bitmask_out
        input_representation_2 = second_graph.bitmask_out
    elif graph_format == GraphFormat.BITMASK_IN:
        input_representation_1 = first_graph.bitmask_in
        input_representation_2 = second_graph.bitmask_in
    else:
        input_representation_1 = first_graph.adjacency_matrix
        input_representation_2 = second_graph.adjacency_matrix
    
    def result(batch_size: int) -> GraphBatch:
        format_representation = np.empty(
            (batch_size, *input_representation_1.shape), dtype=input_representation_1.dtype
        )
        format_representation[:] = input_representation_1
        format_representation[rng.random(size=(batch_size,)) < second_graph_probability] = input_representation_2

        return GraphBatch(
            graph_format=graph_format,
            bitmask_out=format_representation,
            bitmask_in=format_representation,
            adjacency_matrix=format_representation,
            flattened_row_major=format_representation,
            flattened_clockwise=format_representation,
            edge_colors=first_graph.edge_colors,
            is_directed=first_graph.is_directed,
            allow_loops=first_graph.allow_loops,
        )

    return result


def create_edge_perturbation_graph_generator(
    initial_graph: Graph,
    edge_perturbation_probability: float,
    flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
    rng: Optional[np.random.Generator] = None,
):
    """
    Docstring for create_choose_two_graph_generator
    
    :param first_graph: Description
    :type first_graph: Graph
    :param second_graph: Description
    :type second_graph: Graph
    :param second_graph_probability: Description
    :type second_graph_probability: float
    :param graph_format: Description
    """

    if rng is None:
        rng = np.random.default_rng()

    if flattened_ordering == FlattenedOrdering.ROW_MAJOR:
        input_representation = initial_graph.flattened_row_major
    else:
        input_representation = initial_graph.flattened_clockwise
    
    def result(batch_size: int) -> GraphBatch:
        flattened = np.empty(
            (batch_size, input_representation.shape[0]), dtype=np.uint8
        )
        flattened[:] = input_representation

        flattened[rng.random(size=flattened.shape) < edge_perturbation_probability] ^= 1

        return GraphBatch.from_flattened(
            flattened=flattened,
            flattened_ordering=flattened_ordering,
            edge_colors=initial_graph.edge_colors,
            is_directed=initial_graph.is_directed,
            allow_loops=initial_graph.allow_loops,
        )

    return result


def create_random_graph_generator(
    graph_order: int,
    edge_selection_probability: float,
    flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
    edge_colors: int = 2,
    is_directed: bool = False,
    allow_loops: bool = False,
    rng: Optional[np.random.Generator] = None,
):
    """
    Docstring for create_random_graph_generator
    
    :param graph_order: Description
    :type graph_order: int
    :param edge_selection_probability: Description
    :type edge_selection_probability: float
    :param flattened_ordering: Description
    :type flattened_ordering: FlattenedOrdering
    :param rng: Description
    :type rng: Optional[np.random.Generator]
    :param edge_colors: Description
    :type edge_colors: int
    :param is_directed: Description
    :type is_directed: bool
    :param allow_loops: Description
    :type allow_loops: bool
    """

    if rng is None:
        rng = np.random.default_rng()

    # If the graph is directed...
    if is_directed:
        # If loops are allowed, then count all the adjacency matrix entries.
        if allow_loops:
            flattened_length = graph_order * graph_order
        # If loops are not allowed, then count the adjacency matrix entries outside the
        # diagonal.
        else:
            flattened_length = graph_order * (graph_order - 1)
    # If the graph is undirected...
    else:
        # If loops are allowed, then count the entries from the upper triangular part of the
        # adjacency matrix, including the diagonal.
        if allow_loops:
            flattened_length = graph_order * (graph_order + 1) // 2
        # If loops are not allowed, then count the entries from the upper triangular part of
        # the adjacency matrix, excluding the diagonal.
        else:
            flattened_length = graph_order * (graph_order - 1) // 2

    def result(batch_size: int) -> GraphBatch:
        flattened = (rng.random(size=(batch_size, flattened_length)) < edge_selection_probability).astype(np.uint8)

        return GraphBatch.from_flattened(
            flattened=flattened,
            flattened_ordering=flattened_ordering,
            edge_colors=edge_colors,
            is_directed=is_directed,
            allow_loops=allow_loops,
        )

    return result