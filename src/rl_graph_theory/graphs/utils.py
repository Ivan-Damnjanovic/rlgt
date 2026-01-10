"""
This ``Python`` module contains several auxiliary functions concerning the flattened lengths of
$k$-edge-colored looped complete graphs, the conversions between various graph formats, and the
computation of edge (resp. arc) indices with respect to a given edge (resp. arc) ordering.
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
    its order. In other words, the function computes the length of the `numpy.ndarray` list that
    represents the graph structure in the flattened row-major format with color numbers or the
    flattened clockwise format with color numbers.

    :param graph_order: The order of the considered $k$-edge-colored looped complete graph, given
        as a positive `int`.
    :param is_directed: A `bool` that indicates whether the considered graph is a $k$-edge-colored
        looped complete directed graph or a $k$-edge-colored looped complete undirected graph. The
        default value is `False`, i.e., the considered graph is undirected by default.
    :param allow_loops: A `bool` that indicates whether the considered graph is allowed to have
        loops. The default value is `False`, i.e., the considered graph is not allowed to have
        loops by default.

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
    flattened length, i.e., the length of the `numpy.ndarray` list that represents the graph
    structure in the flattened row-major format with color numbers or the flattened clockwise
    format with color numbers.

    :param flattened_length: The flattened length of the considered $k$-edge-colored looped
        complete graph, given as a nonnegative `int`.
    :param is_directed: A `bool` that indicates whether the considered graph is a $k$-edge-colored
        looped complete directed graph or a $k$-edge-colored looped complete undirected graph. The
        default value is `False`, i.e., the considered graph is undirected by default.
    :param allow_loops: A `bool` that indicates whether the considered graph is allowed to have
        loops. The default value is `False`, i.e., the considered graph is not allowed to have
        loops by default.

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
    This function performs a format representation conversion from an adjacency matrix format to
    a flattened format. In other words, it performs the following conversions:

    1. adjacency matrix format with color numbers → flattened row-major format with color numbers;
    2. adjacency matrix format with binary slices → flattened row-major format with binary slices;
    3. adjacency matrix format with color numbers → flattened clockwise format with color numbers;
       and
    4. adjacency matrix format with binary slices → flattened clockwise format with binary slices.

    The function also supports batch-mode operations. More precisely, the input format
    representation may have an arbitrary number of leading dimensions, in which case the function
    applies the operation independently to each (binary or nonbinary) matrix slice defined by the
    last two dimensions, while leaving all the leading dimensions unchanged.

    :param adjacency_matrix: The input adjacency matrix format representation (with color numbers
        or binary slices) that should be converted, given as a `numpy.ndarray` of type
        `numpy.uint8`.
    :param flattened_ordering: An item of the `FlattenedOrdering` enumeration that determines
        whether the conversion should be done to a flattened row-major format or a flattened
        clockwise format.
    :param is_directed: A `bool` that indicates whether the considered graph is a $k$-edge-colored
        looped complete directed graph or a $k$-edge-colored looped complete undirected graph. The
        default value is `False`, i.e., the considered graph is undirected by default.
    :param allow_loops: A `bool` that indicates whether the considered graph is allowed to have
        loops. The default value is `False`, i.e., the considered graph is not allowed to have
        loops by default.

    :return: The output flattened format representation, given as a `numpy.ndarray` of type
        `numpy.uint8`.
    """

    graph_order = adjacency_matrix.shape[-1]
    # Flatten all the leading dimensions.
    temp = adjacency_matrix.reshape(-1, graph_order, graph_order)

    if is_directed:
        # Settle the case where a directed graph is converted to a flattened row-major format.
        if flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            if allow_loops:
                result = temp.reshape(temp.shape[0], -1)
            else:
                result = temp[:, ~np.eye(graph_order, dtype=bool)]

        # Settle the case where a directed graph is converted to a flattened clockwise format.
        else:
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

    # Settle the case where an undirected graph is converted to a flattened (row-major or
    # clockwise) format.
    else:
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

    # Return the conversion result in the required (unflattened) shape.
    return result.reshape(*adjacency_matrix.shape[:-2], -1)


def unflatten_to_adjacency_matrix(
    flattened: np.ndarray,
    flattened_ordering: FlattenedOrdering,
    is_directed: bool = False,
    allow_loops: bool = False,
    graph_order: Optional[int] = None,
) -> np.ndarray:
    """
    This function performs a format representation conversion from a flattened format to an
    adjacency matrix format. In other words, it performs the following conversions:

    1. flattened row-major format with color numbers → adjacency matrix format with color numbers;
    2. flattened row-major format with binary slices → adjacency matrix format with binary slices;
    3. flattened clockwise format with color numbers → adjacency matrix format with color numbers;
       and
    4. flattened clockwise format with binary slices → adjacency matrix format with binary slices.

    The function also supports batch-mode operations. More precisely, the input format
    representation may have an arbitrary number of leading dimensions, in which case the function
    applies the operation independently to each (binary or nonbinary) vector slice defined by the
    last dimension, while leaving all the leading dimensions unchanged.

    :param flattened: The input flattened format representation (with color numbers or binary
        slices) that should be converted, given as a `numpy.ndarray` of type `numpy.uint8`.
    :param flattened_ordering: An item of the `FlattenedOrdering` enumeration that determines
        whether the conversion should be done from a flattened row-major format or a flattened
        clockwise format.
    :param is_directed: A `bool` that indicates whether the considered graph is a $k$-edge-colored
        looped complete directed graph or a $k$-edge-colored looped complete undirected graph. The
        default value is `False`, i.e., the considered graph is undirected by default.
    :param allow_loops: A `bool` that indicates whether the considered graph is allowed to have
        loops. The default value is `False`, i.e., the considered graph is not allowed to have
        loops by default.
    :param graph_order: Either `None`, or a positive `int` that determines the order of the
        considered graph. This argument is optional since the `flattened_length_to_graph_order`
        function can be used to compute the graph order from the flattened length. If the argument
        is `None`, then this is precisely what is done. Otherwise, the provided graph order is
        assumed to be correct and is directly used. The default value is `None`.

    :return: The output adjacency matrix format representation, given as a `numpy.ndarray` of type
        `numpy.uint8`.
    """

    # If the graph order is unknown, then compute it from the flattened length.
    if graph_order is None:
        graph_order = flattened_length_to_graph_order(
            flattened_length=flattened.shape[-1],
            is_directed=is_directed,
            allow_loops=allow_loops,
        )

    # Flatten all the leading dimensions.
    temp = flattened.reshape(-1, flattened.shape[-1])

    # Settle the case when the graphs are directed, with two subcases depending on whether
    # loops are allowed.
    if is_directed:
        # Settle the case where a directed graph is converted to a flattened row-major format.
        if flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            if allow_loops:
                result = temp.reshape(temp.shape[0], graph_order, graph_order)
            else:
                result = np.zeros((temp.shape[0], graph_order, graph_order), dtype=np.uint8)
                result[:, ~np.eye(graph_order, dtype=bool)] = temp

        # Settle the case where a directed graph is converted to a flattened clockwise format.
        else:
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

    # Settle the case where an undirected graph is converted to a flattened (row-major or
    # clockwise) format.
    else:
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

    # Return the conversion result in the required (unflattened) shape.
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

    The function also supports batch-mode operations. More precisely, the input format
    representation may have an arbitrary number of leading dimensions, in which case the function
    applies the operation independently to each matrix slice defined by the last two dimensions or
    vector slice defined by the last dimension, while leaving all the leading dimensions unchanged.

    :param input_representation: The input format representation, given as a `numpy.ndarray` of
        type `numpy.uint8`.
    :param is_flattened_format: A `bool` that determines whether the input format is one of the two
        possible flattened formats, i.e., the flattened row-major format with color numbers or the
        flattened clockwise format with color numbers.
    :param edge_colors: A positive `int` (not below 2) that represents the number of proper edge
        colors, i.e., $k$, in the considered $k$-edge-colored looped complete graph. The default
        value is 2.
    :param allow_loops: A `bool` that indicates whether the considered graph is allowed to have
        loops. The default value is `False`, i.e., the considered graph is not allowed to have
        loops by default.

    :return: The corresponding output format representation, given as a `numpy.ndarray` of type
        `numpy.uint8`.

    :note: The output format representation is given in the corresponding reduced format if
        possible, i.e., if the graphs from all the slices are fully colored.
    """

    # Determine whether the graphs from all the slices are fully colored.
    is_fully_colored = np.max(input_representation) < edge_colors
    color_indices = np.arange(edge_colors, dtype=np.uint8)

    # If possible, use a reduced format.
    if is_fully_colored:
        if is_flattened_format:
            result = (
                np.expand_dims(input_representation, axis=-2) == color_indices[1:, None]
            ).astype(np.uint8)
        else:
            result = (
                np.expand_dims(input_representation, axis=-3) == color_indices[1:, None, None]
            ).astype(np.uint8)
    # Otherwise, use a standard (non-reduced) format.
    else:
        if is_flattened_format:
            result = (
                np.expand_dims(input_representation, axis=-2) == color_indices[:, None]
            ).astype(np.uint8)
        else:
            result = (
                np.expand_dims(input_representation, axis=-3) == color_indices[:, None, None]
            ).astype(np.uint8)

        # If the output format is the adjacency matrix format with binary slices, and loops are not
        # allowed, then all the diagonal entries corresponding the color 0 must be set to 0.
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

    The function also supports batch-mode operations. More precisely, the input format
    representation may have an arbitrary number of leading dimensions, in which case the function
    applies the operation independently to each matrix slice defined by the last two dimensions or
    vector slice defined by the last dimension, while leaving all the leading dimensions unchanged.

    :param input_representation: The input format representation, given as a `numpy.ndarray` of
        type `numpy.uint8`.
    :param is_flattened_format: A `bool` that determines whether the input format is one of the two
        possible flattened formats, i.e., the flattened row-major format with binary slices or the
        flattened clockwise format with binary slices.
    :param edge_colors: A positive `int` (not below 2) that represents the number of proper edge
        colors, i.e., $k$, in the considered $k$-edge-colored looped complete graph. The default
        value is 2.
    :param allow_loops: A `bool` that indicates whether the considered graph is allowed to have
        loops. The default value is `False`, i.e., the considered graph is not allowed to have
        loops by default.

    :return: The corresponding output format representation, given as a `numpy.ndarray` of type
        `numpy.uint8`.

    :note: The input format representation can be given in both the corresponding reduced format
        and the corresponding standard (non-reduced) format.
    """

    # Determine whether the input format representation is given in the corresponding reduced
    # format.
    if is_flattened_format:
        is_reduced_format = input_representation.shape[-2] == edge_colors - 1
    else:
        is_reduced_format = input_representation.shape[-3] == edge_colors - 1

    # Settle the case where a reduced format is being used as the input format.
    if is_reduced_format:
        weights = np.arange(1, edge_colors, dtype=np.uint8)
        if is_flattened_format:
            result = np.sum(input_representation * weights[:, None], axis=-2)
        else:
            result = np.sum(input_representation * weights[:, None, None], axis=-3)

    # Settle the case where a standard (non-reduced) format is being used as the input format.
    else:
        # Determine the output format representation shape.
        if is_flattened_format:
            output_shape = input_representation.shape[:-2] + input_representation.shape[-1:]
        else:
            output_shape = input_representation.shape[:-3] + input_representation.shape[-2:]

        result = np.full(output_shape, edge_colors, dtype=np.uint8)
        # If the output format is the adjacency matrix format with color numbers, and loops are not
        # allowed, then all the diagonal entries must be set to 0.
        if not is_flattened_format and not allow_loops:
            indices = np.arange(result.shape[-1])
            result[..., indices, indices] = 0

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
    This function considers a $k$-edge-colored looped complete graph, and computes the index of
    each of the edges (resp. arcs) from a given list, with respect to the row-major or clockwise
    ordering, as described in the `FlattenedOrdering` enumeration. The edges (resp. arcs) are given
    as ordered pairs of vertices consisting of the starting vertex and the ending vertex. It is
    possible to configure which of the two edge (resp. arc) orderings should be used to arrange the
    edges (resp. arcs), as well as select the graph order and choose whether the graphs should be
    directed or undirected, and whether loops should be allowed. In the case of undirected graphs,
    it does not matter which vertex is the starting one and which is the ending. If loops are not
    allowed, then the starting and the ending vertex from each pair must be distinct.

    :param graph_order: A positive `int` that determines the order of the considered graph.
    :param starting_vertices: A `numpy.ndarray` list of type `numpy.int32` that contains the
        starting vertex from each of the given ordered pairs of vertices that represent the edges
        (resp. arcs).
    :param ending_vertices: A `numpy.ndarray` list of type `numpy.int32` that contains the ending
        vertex from each of the given ordered pairs of vertices that represent the edges (resp.
        arcs).
    :param flattened_ordering: An item of the `FlattenedOrdering` enumeration that determines
        whether the edges (resp. arcs) should be arranged in the row-major order or the clockwise
        order. The default value is `FlattenedOrdering.ROW_MAJOR`, i.e., the edges (resp. arcs)
        should be arranged in the row-major order by default.
    :param is_directed: A `bool` that indicates whether the considered graph is a $k$-edge-colored
        looped complete directed graph or a $k$-edge-colored looped complete undirected graph. The
        default value is `False`, i.e., the considered graph is undirected by default.
    :param allow_loops: A `bool` that indicates whether the considered graph is allowed to have
        loops. The default value is `False`, i.e., the considered graph is not allowed to have
        loops by default.

    :return: The computed edge (resp. arc) indices, given in the natural order through a
        `numpy.ndarray` list of type `numpy.int32`.
    """

    if is_directed:
        # Settle the case for the directed graphs with the row-major edge (resp. arc) ordering.
        if flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            if allow_loops:
                result = starting_vertices * graph_order + ending_vertices
            else:
                result = (
                    starting_vertices * (graph_order - 1)
                    + ending_vertices
                    - (ending_vertices >= starting_vertices).astype(np.int32)
                )
        # Settle the case for the directed graphs with the clockwise edge (resp. arc) ordering.
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
