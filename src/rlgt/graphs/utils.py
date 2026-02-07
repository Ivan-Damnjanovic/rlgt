"""
This ``Python`` module provides auxiliary functions for working with $k$-edge-colored looped
complete graphs. The functions include computing flattened lengths, converting between various
graph formats, and calculating edge (resp. arc) indices according to a specified edge (resp. arc)
ordering.
"""

from math import isqrt
from typing import Optional

import numpy as np

from .graph_formats import FlattenedOrdering


def graph_order_to_flattened_length(
    graph_order: int,
    is_directed: bool = False,
    allow_loops: bool = False,
) -> int:
    """
    This function computes the flattened length of a $k$-edge-colored looped complete graph given
    its order. The flattened length is the number of entries in the `numpy.ndarray` vector that
    represents the graph structure in either the flattened row-major format with color numbers
    or the flattened clockwise format with color numbers.

    :param graph_order: The order of the considered $k$-edge-colored looped complete graph, given
        as a positive `int`.
    :param is_directed: A `bool` indicating whether the considered graph is directed. The default
        value is `False`, i.e., the graph is undirected by default.
    :param allow_loops: A `bool` indicating whether the considered graph is allowed to have loops.
        The default value is `False`, i.e., loops are not allowed by default.

    :return: The computed flattened length, given as a nonnegative `int`.
    """

    if is_directed:
        if allow_loops:
            flattened_length = graph_order * graph_order
        else:
            flattened_length = graph_order * (graph_order - 1)
    else:
        if allow_loops:
            flattened_length = graph_order * (graph_order + 1) // 2
        else:
            flattened_length = graph_order * (graph_order - 1) // 2

    return flattened_length


def flattened_length_to_graph_order(
    flattened_length: int,
    is_directed: bool = False,
    allow_loops: bool = False,
) -> int:
    """
    This function computes the order of a $k$-edge-colored looped complete graph given its
    flattened length. The flattened length is the number of entries in the `numpy.ndarray` vector
    that represents the graph structure in either the flattened row-major format with color numbers
    or the flattened clockwise format with color numbers.

    :param flattened_length: The flattened length of the considered graph, given as a nonnegative
        `int`.
    :param is_directed: A `bool` indicating whether the considered graph is directed. The default
        value is `False`, i.e., the graph is undirected by default.
    :param allow_loops: A `bool` indicating whether the considered graph is allowed to have loops.
        The default value is `False`, i.e., loops are not allowed by default.

    :return: The computed graph order, given as a positive `int`.
    """

    if is_directed:
        if allow_loops:
            # Given $n^2$, find $n$.
            graph_order = isqrt(flattened_length)
        else:
            # Given $n^2 - n$, find $n$.
            graph_order = (isqrt(4 * flattened_length + 1) + 1) // 2
    else:
        if allow_loops:
            # Given \binom{n + 1}{2}$, find $n$.
            graph_order = (isqrt(8 * flattened_length + 1) - 1) // 2
        else:
            # Given $\binom{n}{2}$, find $n$.
            graph_order = (isqrt(8 * flattened_length + 1) + 1) // 2

    return graph_order


def flatten_from_adjacency_matrix(
    adjacency_matrix: np.ndarray,
    flattened_ordering: FlattenedOrdering,
    is_directed: bool = False,
    allow_loops: bool = False,
) -> np.ndarray:
    """
    This function converts a graph or batch of graphs from an adjacency matrix format to a
    flattened format. In other words, it performs one of the following conversions depending on the
    input:

    1. adjacency matrix format with color numbers → flattened row-major format with color numbers;
    2. adjacency matrix format with binary slices → flattened row-major format with binary slices;
    3. adjacency matrix format with color numbers → flattened clockwise format with color numbers;
       and
    4. adjacency matrix format with binary slices → flattened clockwise format with binary slices.

    The function also supports batch-mode operations. Specifically, the input may have an arbitrary
    number of leading dimensions, in which case the conversion is applied independently to each
    matrix slice defined by the last two dimensions, while all leading dimensions remain unchanged.

    :param adjacency_matrix: The adjacency matrix representation of the graph or batch of graphs
        (with color numbers or binary slices), given as a `numpy.ndarray` of type `numpy.uint8`.
    :param flattened_ordering: An item of the `FlattenedOrdering` enumeration that specifies
        whether the output should use a row-major or clockwise edge (resp. arc) ordering.
    :param is_directed: A `bool` indicating whether the graph or each graph in the batch is
        directed. The default value is `False`, i.e., the graphs are undirected by default.
    :param allow_loops: A `bool` indicating whether loops are allowed in the graph or each graph in
        the batch. The default value is `False`, i.e., loops are not allowed by default.

    :return: The flattened format representation of the graph or batch of graphs, given as a
        `numpy.ndarray` of type `numpy.uint8`. The output shape preserves the leading dimensions
        of the input, with the last two dimensions replaced by the flattened length.
    """

    graph_order = adjacency_matrix.shape[-1]
    # Reshape to isolate individual adjacency matrices, preserving any leading batch dimensions.
    temp = adjacency_matrix.reshape(-1, graph_order, graph_order)

    if is_directed:
        if flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            # Flatten each directed adjacency matrix in row-major order.
            if allow_loops:
                result = temp.reshape(temp.shape[0], -1)
            else:
                # Skip the diagonal entries if loops are not allowed.
                result = temp[:, ~np.eye(graph_order, dtype=bool)]

        else:
            # Flatten each directed adjacency matrix in clockwise order.
            if allow_loops:
                result = np.zeros((temp.shape[0], graph_order * graph_order), dtype=np.uint8)
                result[:, 0] = temp[:, 0, 0]

                start = 1
                for layer in range(1, graph_order):
                    result[:, start : start + layer + 1] = temp[:, : layer + 1, layer]
                    start += layer + 1

                    result[:, start : start + layer] = temp[:, layer, layer - 1 :: -1]
                    start += layer

            else:
                result = np.zeros((temp.shape[0], graph_order * (graph_order - 1)), dtype=np.uint8)

                start = 0
                for layer in range(1, graph_order):
                    result[:, start : start + layer] = temp[:, :layer, layer]
                    start += layer

                    result[:, start : start + layer] = temp[:, layer, layer - 1 :: -1]
                    start += layer

    else:
        # Flatten each undirected adjacency matrix, choosing upper or lower triangular indices
        # depending on the edge ordering and whether loops are allowed.
        if flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            if allow_loops:
                rows, columns = np.triu_indices(graph_order, k=0)
            else:
                rows, columns = np.triu_indices(graph_order, k=1)
        else:
            if allow_loops:
                rows, columns = np.tril_indices(graph_order, k=0)
            else:
                rows, columns = np.tril_indices(graph_order, k=-1)

        result = temp[:, rows, columns]

    # Restore the original batch dimensions and return the flattened representation.
    return result.reshape(*adjacency_matrix.shape[:-2], -1)


def unflatten_to_adjacency_matrix(
    flattened: np.ndarray,
    flattened_ordering: FlattenedOrdering,
    is_directed: bool = False,
    allow_loops: bool = False,
    graph_order: Optional[int] = None,
) -> np.ndarray:
    """
    This function converts a graph or a batch of graphs from a flattened format to an adjacency
    matrix format. Specifically, it performs one of the following conversions depending on the
    input:

    1. flattened row-major format with color numbers → adjacency matrix format with color numbers;
    2. flattened row-major format with binary slices → adjacency matrix format with binary slices;
    3. flattened clockwise format with color numbers → adjacency matrix format with color numbers;
       and
    4. flattened clockwise format with binary slices → adjacency matrix format with binary slices.

    The function supports batch-mode operations. If the input has leading dimensions, the
    conversion is applied independently to each vector slice defined by the last dimension, while
    preserving the leading dimensions.

    :param flattened: The input flattened format representation (with color numbers or binary
        slices) for the graph or batch of graphs, given as a `numpy.ndarray` of type `numpy.uint8`.
    :param flattened_ordering: An item of the `FlattenedOrdering` enumeration that specifies
        whether the input uses a flattened or clockwise edge (resp. arc) ordering.
    :param is_directed: A `bool` indicating whether the graph or each graph in the batch is
        directed. The default value is `False`, i.e., the graphs are undirected by default.
    :param allow_loops: A `bool` indicating whether loops are allowed in the graph or each graph in
        the batch. The default value is `False`, i.e., loops are not allowed by default.
    :param graph_order: Either `None` or a positive `int` specifying the order of the graph(s). If
        `None`, then the graph order is computed from the flattened length using
        `flattened_length_to_graph_order`. Otherwise, the provided value is used directly. The
        default value is `None`.

    :return: The adjacency matrix representation of the graph or batch of graphs, given as a
        `numpy.ndarray` of type `numpy.uint8`. The output preserves any leading dimensions and has
        the last two dimensions equal to the graph order.
    """

    # Determine the graph order if not provided.
    if graph_order is None:
        graph_order = flattened_length_to_graph_order(
            flattened_length=flattened.shape[-1],
            is_directed=is_directed,
            allow_loops=allow_loops,
        )

    # Flatten all leading dimensions to simplify the processing of individual graphs.
    temp = flattened.reshape(-1, flattened.shape[-1])

    if is_directed:
        if flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            # Restore each directed graph from a flattened row-major vector.
            if allow_loops:
                result = temp.reshape(temp.shape[0], graph_order, graph_order)
            else:
                result = np.zeros((temp.shape[0], graph_order, graph_order), dtype=np.uint8)
                result[:, ~np.eye(graph_order, dtype=bool)] = temp

        else:
            # Restore each directed graph from a flattened clockwise vector.
            result = np.zeros((temp.shape[0], graph_order, graph_order), dtype=np.uint8)

            if allow_loops:
                result[:, 0, 0] = temp[:, 0]

                start = 1
                for layer in range(1, graph_order):
                    result[:, : layer + 1, layer] = temp[:, start : start + layer + 1]
                    start += layer + 1

                    result[:, layer, :layer] = temp[:, start + layer - 1 : start - 1 : -1]
                    start += layer

            else:
                start = 0
                for layer in range(1, graph_order):
                    result[:, :layer, layer] = temp[:, start : start + layer]
                    start += layer

                    result[:, layer, :layer] = temp[:, start + layer - 1 : start - 1 : -1]
                    start += layer

    else:
        # Restore each undirected graph from a flattened format vector (row-major or clockwise).
        result = np.zeros((temp.shape[0], graph_order, graph_order), dtype=np.uint8)

        if flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            if allow_loops:
                rows, columns = np.triu_indices(graph_order, k=0)
            else:
                rows, columns = np.triu_indices(graph_order, k=1)
        else:
            if allow_loops:
                rows, columns = np.tril_indices(graph_order, k=0)
            else:
                rows, columns = np.tril_indices(graph_order, k=-1)

        result[:, rows, columns] = temp
        result[:, columns, rows] = temp

    # Reshape to restore the original batch dimensions.
    return result.reshape(*flattened.shape[:-1], graph_order, graph_order)


def color_numbers_to_binary_slices(
    input_representation: np.ndarray,
    is_flattened_format: bool,
    edge_colors: int = 2,
    allow_loops: bool = False,
) -> np.ndarray:
    """
    This function performs a format representation conversion from a graph format with color
    numbers to the corresponding format with binary slices. In other words, it performs the
    following conversions:

    1. adjacency matrix format with color numbers → adjacency matrix format with binary slices;
    2. flattened row-major format with color numbers → flattened row-major format with binary
       slices; and
    3. flattened clockwise format with color numbers → flattened clockwise format with binary
       slices.

    The function also supports batch-mode operations. If the input has leading dimensions, the
    conversion is applied independently to each matrix slice defined by the last two dimensions
    (for adjacency matrices) or vector slice defined by the last dimension (for flattened formats),
    while preserving all leading dimensions.

    :param input_representation: The input format representation, given as a `numpy.ndarray` of
        type `numpy.uint8`.
    :param is_flattened_format: A `bool` that determines whether the input format is one of the two
        possible flattened formats, i.e., the flattened row-major format with color numbers or the
        flattened clockwise format with color numbers.
    :param edge_colors: A positive `int` (not below 2) that represents the number of proper edge
        colors, i.e., $k$, in the considered graph or batch of graphs. The default value is 2.
    :param allow_loops: A `bool` that indicates whether the considered graph or batch of graphs
        is allowed to have loops. The default value is `False`.

    :return: The corresponding output format representation, given as a `numpy.ndarray` of type
        `numpy.uint8`.

    :note: The output is given in the corresponding reduced format with binary slices whenever
        possible, i.e., when all the considered graphs are fully colored.
    """

    # Determine whether all graphs are fully colored.
    is_fully_colored = np.max(input_representation) < edge_colors
    color_indices = np.arange(edge_colors, dtype=np.uint8)

    if is_fully_colored:
        # Use the reduced binary slice representation (omit the slice corresponding to color 0).
        if is_flattened_format:
            result = (
                np.expand_dims(input_representation, axis=-2) == color_indices[1:, None]
            ).astype(np.uint8)
        else:
            result = (
                np.expand_dims(input_representation, axis=-3) == color_indices[1:, None, None]
            ).astype(np.uint8)
    else:
        # Use the standard binary slice representation with all colors included.
        if is_flattened_format:
            result = (
                np.expand_dims(input_representation, axis=-2) == color_indices[:, None]
            ).astype(np.uint8)
        else:
            result = (
                np.expand_dims(input_representation, axis=-3) == color_indices[:, None, None]
            ).astype(np.uint8)

        # If the output format is the adjacency matrix format with binary slices, and loops are not
        # allowed, zero out the diagonal for color 0.
        if not is_flattened_format and not allow_loops:
            indices = np.arange(result.shape[-1])
            result[..., 0, indices, indices] = 0

    return result


def binary_slices_to_color_numbers(
    input_representation: np.ndarray,
    is_flattened_format: bool,
    edge_colors: int = 2,
    allow_loops: bool = False,
) -> np.ndarray:
    """
    This function performs a format representation conversion from a graph format with binary
    slices to the corresponding format with color numbers. In other words, it performs the
    following conversions:

    1. adjacency matrix format with binary slices → adjacency matrix format with color numbers;
    2. flattened row-major format with binary slices → flattened row-major format with color
       numbers; and
    3. flattened clockwise format with binary slices → flattened clockwise format with color
       numbers.

    The function also supports batch-mode operations. If the input contains multiple graphs
    arranged along leading dimensions, the conversion is applied independently to each graph, while
    preserving all leading dimensions.

    :param input_representation: The input format representation, given as a `numpy.ndarray` of
        type `numpy.uint8`.
    :param is_flattened_format: A `bool` that determines whether the input format is one of the two
        possible flattened formats, i.e., the flattened row-major format with color numbers or the
        flattened clockwise format with color numbers.
    :param edge_colors: A positive `int` (not below 2) that represents the number of proper edge
        colors, i.e., $k$, in the considered graph or batch of graphs. The default value is 2.
    :param allow_loops: A `bool` that indicates whether the considered graph or batch of graphs
        is allowed to have loops. The default value is `False`.

    :return: The corresponding output format representation, given as a `numpy.ndarray` of type
        `numpy.uint8`.

    :note: The input format representation can be given in either the reduced binary slice format
        or the standard (non-reduced) format.
    """

    # Determine whether the input uses a reduced binary slice format.
    if is_flattened_format:
        is_reduced_format = input_representation.shape[-2] == edge_colors - 1
    else:
        is_reduced_format = input_representation.shape[-3] == edge_colors - 1

    if is_reduced_format:
        # Use color weights starting from 1 because color 0 is omitted.
        weights = np.arange(1, edge_colors, dtype=np.uint8)
        if is_flattened_format:
            result = np.sum(input_representation * weights[:, None], axis=-2)
        else:
            result = np.sum(input_representation * weights[:, None, None], axis=-3)

    # Settle the case where a standard (non-reduced) format is being used as the input format.
    else:
        # Initialize the output with the default color for uncolored edges.
        if is_flattened_format:
            output_shape = input_representation.shape[:-2] + input_representation.shape[-1:]
        else:
            output_shape = input_representation.shape[:-3] + input_representation.shape[-2:]

        result = np.full(output_shape, edge_colors, dtype=np.uint8)

        # If the output format is the adjacency matrix format with color numbers, and loops are not
        # allowed, zero out the diagonal.
        if not is_flattened_format and not allow_loops:
            indices = np.arange(result.shape[-1])
            result[..., indices, indices] = 0

        # Apply weights to subtract the contributions of each binary slice.
        weights = np.arange(edge_colors, 0, -1, dtype=np.uint8)
        if is_flattened_format:
            result -= np.sum(input_representation * weights[:, None], axis=-2)
        else:
            result -= np.sum(input_representation * weights[:, None, None], axis=-3)

    return result


def compute_edge_indices(
    graph_order: int,
    starting_vertices: np.ndarray,
    ending_vertices: np.ndarray,
    flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
    is_directed: bool = False,
    allow_loops: bool = False,
) -> np.ndarray:
    """
    This function considers a $k$-edge-colored looped complete graph and computes the index of each
    edge (resp. arc) from a given list, with respect to the row-major or clockwise ordering as
    described in the `FlattenedOrdering` enumeration. The edges (resp. arcs) are given as ordered
    pairs of vertices consisting of the starting vertex and the ending vertex. The user can select
    the edge (resp. arc) ordering, the graph order, whether the graphs are directed or undirected,
    and whether loops are allowed. For undirected graphs, the starting and ending vertices of a
    pair are interchangeable. If loops are not allowed, then the starting and the ending vertex
    from each pair must be distinct.

    :param graph_order: A positive `int` that specifies the order of the considered graph.
    :param starting_vertices: A `numpy.ndarray` of type `numpy.int32` containing the starting
        vertices of the edges (resp. arcs) whose indices are to be computed.
    :param ending_vertices: A `numpy.ndarray` of type `numpy.int32` containing the ending
        vertices of the edges (resp. arcs) whose indices are to be computed.
    :param flattened_ordering: An item of the `FlattenedOrdering` enumeration that specifies
        whether the edges (resp. arcs) should be arranged in row-major order or clockwise order.
        The default value is `FlattenedOrdering.ROW_MAJOR`.
    :param is_directed: A `bool` that indicates whether the graph is directed. The default value is
        `False`, i.e., the graph is undirected by default.
    :param allow_loops: A `bool` that indicates whether loops are allowed in the graph. The default
        value is `False`.

    :return: A `numpy.ndarray` of type `numpy.int32` containing the indices of the edges (resp.
        arcs) with respect to the selected edge (resp. arc) ordering.
    """

    if is_directed:
        # Settle the case for the directed graphs with row-major edge (resp. arc) ordering.
        if flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            if allow_loops:
                result = starting_vertices * graph_order + ending_vertices
            else:
                result = (
                    starting_vertices * (graph_order - 1)
                    + ending_vertices
                    - (ending_vertices >= starting_vertices).astype(np.int32)
                )
        # Settle the case for the directed graphs with clockwise edge (resp. arc) ordering.
        else:
            layer = np.maximum(starting_vertices, ending_vertices)

            if allow_loops:
                result = layer * layer + layer - ending_vertices + starting_vertices
            else:
                result = (
                    layer * layer
                    - ending_vertices
                    + starting_vertices
                    - (ending_vertices <= starting_vertices).astype(np.int32)
                )

    # Settle the case for the undirected graphs.
    else:
        rows = np.minimum(starting_vertices, ending_vertices)
        columns = np.maximum(starting_vertices, ending_vertices)

        if flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            if allow_loops:
                result = rows * (2 * graph_order - 1 - rows) // 2 + columns
            else:
                result = rows * (2 * graph_order - 3 - rows) // 2 + columns - 1
        else:
            if allow_loops:
                result = columns * (columns + 1) // 2 + rows
            else:
                result = columns * (columns - 1) // 2 + rows

    return result
