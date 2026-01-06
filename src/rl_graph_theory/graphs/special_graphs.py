"""
This ``Python`` module contains various classes that inherit from the `Graph` class and are used to
construct $k$-edge-colored looped complete graphs with some particular structure.
"""

from typing import List, Set

import numpy as np

from .graph import Graph
from .graph_formats import GraphFormat
from .utils import graph_order_to_flattened_length


class MonochromaticGraph(Graph):
    """
    This class inherits from the `Graph` class and it is used to instantiate monochromatic graphs,
    i.e., $k$-edge-colored looped complete graphs where all the edges (resp. arcs) are colored with
    the same color (or they are all uncolored). It is also possible to configure which starting
    graph formats an instance should get initialized in.
    """

    def __init__(
        self,
        graph_formats: Set[GraphFormat],
        graph_order: int,
        edge_colors: int = 2,
        selected_color: int = 0,
        is_directed: bool = False,
        allow_loops: bool = False,
    ):
        """
        This constructor initializes the desired monochromatic graph in the selected graph formats.

        :param graph_formats: A nonempty set containing items of the `GraphFormat` enumeration
            whose elements determine the starting graph formats that the graph should get
            initialized in.
        :param graph_order: The graph order, given as a positive `int`.
        :param edge_colors: A positive `int` (not below 2) that represents the number of proper
            edge colors, i.e., $k$. The default value is 2.
        :param selected_color: The edge color that all the edges (resp. arcs) should get colored
            with, given as a nonnegative `int` between 0 and ``edge_colors``. If this argument
            equals ``edge_colors``, then this means that all the edges (resp. arcs) should be
            uncolored. The default value is 0.
        :param is_directed: A `bool` that indicates whether the given monochromatic graph is a
            $k$-edge-colored looped complete directed graph or a $k$-edge-colored looped complete
            undirected graph. The default value is `False`, i.e., the given monochromatic graph is
            undirected by default.
        :param allow_loops: A `bool` that indicates whether the given monochromatic graph is
            allowed to have loops. The default value is `False`, i.e., the given monochromatic
            graph is not allowed to have loops by default.
        """

        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None
        flattened_colors = None
        flattened_binary = None

        # The graph is not fully colored if the selected edge color equals ``edge_colors``, and the
        # number of edges (resp. arcs) is at least one (which comes down to having loops allowed or
        # having at least two vertices).
        not_fully_colored = selected_color != edge_colors and (allow_loops or graph_order >= 2)
        # Determine the first dimension in the bitmask formats and the formats with binary slices.
        # Use the reduced formats if possible.
        if not_fully_colored:
            color_dim = edge_colors
        else:
            color_dim = edge_colors - 1

        flattened_length = graph_order_to_flattened_length(
            graph_order=graph_order,
            is_directed=is_directed,
            allow_loops=allow_loops,
        )

        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            bitmask = np.zeros((color_dim, graph_order), dtype=np.uint64)

            if selected_color != 0 and selected_color != edge_colors:
                bitmask[selected_color - 1, :] = (1 << graph_order) - 1
                # Remove the loops if they are not allowed.
                if not allow_loops:
                    bitmask[selected_color - 1, :] -= 1 << np.arange(graph_order, dtype=np.uint64)

        if GraphFormat.ADJACENCY_MATRIX_COLORS in graph_formats:
            adjacency_matrix_colors = np.full(
                (graph_order, graph_order), selected_color, dtype=np.uint8
            )
            # Remove the loops if they are not allowed.
            if not allow_loops and selected_color != 0:
                np.fill_diagonal(adjacency_matrix_colors, 0)

        if GraphFormat.ADJACENCY_MATRIX_BINARY in graph_formats:
            adjacency_matrix_binary = np.zeros(
                (color_dim, graph_order, graph_order), dtype=np.uint8
            )

            if selected_color != 0 and selected_color != edge_colors:
                adjacency_matrix_binary[selected_color] = 1
                # Remove the loops if they are not allowed.
                if not allow_loops:
                    np.fill_diagonal(adjacency_matrix_binary[selected_color], 0)

        if graph_formats & {
            GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
            GraphFormat.FLATTENED_CLOCKWISE_COLORS,
        }:
            flattened_colors = np.full((flattened_length), selected_color, dtype=np.uint8)

        if graph_formats & {
            GraphFormat.FLATTENED_ROW_MAJOR_BINARY,
            GraphFormat.FLATTENED_CLOCKWISE_BINARY,
        }:
            flattened_binary = np.zeros((color_dim, flattened_length), dtype=np.uint8)
            if selected_color != 0 and selected_color != edge_colors:
                flattened_binary[selected_color] = 1

        super().__init__(
            edge_colors=edge_colors,
            is_directed=is_directed,
            allow_loops=allow_loops,
            bitmask_out=bitmask,
            bitmask_in=bitmask,
            adjacency_matrix_colors=adjacency_matrix_colors,
            adjacency_matrix_binary=adjacency_matrix_binary,
            flattened_row_major_colors=flattened_colors,
            flattened_row_major_binary=flattened_binary,
            flattened_clockwise_colors=flattened_colors,
            flattened_clockwise_binary=flattened_binary,
        )


# Although an empty graph can conceptually be regarded as a monochromatic graph, this class
# directly inherits from the `Graph` class instead of the `MonochromaticGraph` class for
# implementational reasons.
class EmptyGraph(Graph):
    """
    This class inherits from the `Graph` class and it is used to instantiate empty graphs. In other
    words, it produces a 2-edge-colored loopless complete undirected graph of a given order such
    that all of its edges are colored with the color 0. It is also possible to configure which
    starting graph formats an instance should get initialized in.
    """

    def __init__(self, graph_formats: Set[GraphFormat], graph_order: int):
        """
        This constructor initializes the desired empty graph in the selected graph formats.

        :param graph_formats: A nonempty set containing items of the `GraphFormat` enumeration
            whose elements determine the starting graph formats that the graph should get
            initialized in.
        :param graph_order: The graph order, given as a positive `int`.
        """

        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None
        flattened_colors = None
        flattened_binary = None

        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            bitmask = np.zeros((1, graph_order), dtype=np.uint64)

        if graph_formats & {
            GraphFormat.ADJACENCY_MATRIX_COLORS,
            GraphFormat.ADJACENCY_MATRIX_BINARY,
        }:
            adjacency_matrix_colors = np.zeros((graph_order, graph_order), dtype=np.uint8)
            adjacency_matrix_binary = np.expand_dims(adjacency_matrix_colors, axis=0)

        if graph_formats & {
            GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
            GraphFormat.FLATTENED_ROW_MAJOR_BINARY,
            GraphFormat.FLATTENED_CLOCKWISE_COLORS,
            GraphFormat.FLATTENED_CLOCKWISE_BINARY,
        }:
            flattened_colors = np.zeros((graph_order * (graph_order - 1) // 2,), dtype=np.uint8)
            flattened_binary = np.expand_dims(flattened_colors, axis=0)

        super().__init__(
            bitmask_out=bitmask,
            bitmask_in=bitmask,
            adjacency_matrix_colors=adjacency_matrix_colors,
            adjacency_matrix_binary=adjacency_matrix_binary,
            flattened_row_major_colors=flattened_colors,
            flattened_row_major_binary=flattened_binary,
            flattened_clockwise_colors=flattened_colors,
            flattened_clockwise_binary=flattened_binary,
        )


# Although a complete graph can conceptually be regarded as a monochromatic graph, this class
# directly inherits from the `Graph` class instead of the `MonochromaticGraph` class for
# implementational reasons.
class CompleteGraph(Graph):
    """
    This class inherits from the `Graph` class and it is used to instantiate complete graphs. In
    other words, it produces a 2-edge-colored loopless complete undirected graph of a given order
    such that all of its edges are colored with the color 1. It is also possible to configure which
    starting graph formats an instance should get initialized in.
    """

    def __init__(self, graph_formats: Set[GraphFormat], graph_order: int):
        """
        This constructor initializes the desired complete graph in the selected graph formats.

        :param graph_formats: A nonempty set containing items of the `GraphFormat` enumeration
            whose elements determine the starting graph formats that the graph should get
            initialized in.
        :param graph_order: The graph order, given as a positive `int`.
        """

        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None
        flattened_colors = None
        flattened_binary = None

        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            bitmask = np.full((1, graph_order), (1 << graph_order) - 1, dtype=np.uint64)
            bitmask[0, :] -= 1 << np.arange(graph_order, dtype=np.uint64)

        if graph_formats & {
            GraphFormat.ADJACENCY_MATRIX_COLORS,
            GraphFormat.ADJACENCY_MATRIX_BINARY,
        }:
            adjacency_matrix_colors = np.ones((graph_order, graph_order), dtype=np.uint8)
            np.fill_diagonal(adjacency_matrix_colors, 0)
            adjacency_matrix_binary = np.expand_dims(adjacency_matrix_colors, axis=0)

        if graph_formats & {
            GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
            GraphFormat.FLATTENED_ROW_MAJOR_BINARY,
            GraphFormat.FLATTENED_CLOCKWISE_COLORS,
            GraphFormat.FLATTENED_CLOCKWISE_BINARY,
        }:
            flattened_colors = np.ones((graph_order * (graph_order - 1) // 2,), dtype=np.uint8)
            flattened_binary = np.expand_dims(flattened_colors, axis=0)

        super().__init__(
            bitmask_out=bitmask,
            bitmask_in=bitmask,
            adjacency_matrix_colors=adjacency_matrix_colors,
            adjacency_matrix_binary=adjacency_matrix_binary,
            flattened_row_major_colors=flattened_colors,
            flattened_row_major_binary=flattened_binary,
            flattened_clockwise_colors=flattened_colors,
            flattened_clockwise_binary=flattened_binary,
        )


class AlmostCompleteGraph(Graph):
    r"""
    This class inherits from the `Graph` class and it is used to instantiate almost-complete
    graphs. In other words, it produces a 2-edge-colored loopless complete undirected graph of a
    given order $n$ such that all of its edges are colored with the color 1, except for the edge
    between the vertices $n - 2$ and $n - 1$, which is colored with the color 0. It is also
    possible to configure which starting graph formats an instance should get initialized in.
    """

    def __init__(self, graph_formats: Set[GraphFormat], graph_order: int):
        """
        This constructor initializes the desired almost-complete graph in the selected graph
        formats.

        :param graph_formats: A nonempty set containing items of the `GraphFormat` enumeration
            whose elements determine the starting graph formats that the graph should get
            initialized in.
        :param graph_order: The graph order, given as a positive `int`.
        """

        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None
        flattened_colors = None
        flattened_binary = None

        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            bitmask = np.full((1, graph_order), (1 << graph_order) - 1, dtype=np.uint64)
            bitmask[0, :] -= 1 << np.arange(graph_order, dtype=np.uint64)
            # Remove the edge between the vertices $n - 2$ and $n - 1$.
            bitmask[0, -1] -= 1 << (graph_order - 2)
            bitmask[0, -2] -= 1 << (graph_order - 1)

        if graph_formats & {
            GraphFormat.ADJACENCY_MATRIX_COLORS,
            GraphFormat.ADJACENCY_MATRIX_BINARY,
        }:
            adjacency_matrix_colors = np.ones((graph_order, graph_order), dtype=np.uint8)
            np.fill_diagonal(adjacency_matrix_colors, 0)
            # Remove the edge between the vertices $n - 2$ and $n - 1$.
            adjacency_matrix_colors[-1, -2] = 0
            adjacency_matrix_colors[-2, -1] = 0

            adjacency_matrix_binary = np.expand_dims(adjacency_matrix_colors, axis=0)

        if graph_formats & {
            GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
            GraphFormat.FLATTENED_ROW_MAJOR_BINARY,
            GraphFormat.FLATTENED_CLOCKWISE_COLORS,
            GraphFormat.FLATTENED_CLOCKWISE_BINARY,
        }:
            flattened_colors = np.ones((graph_order * (graph_order - 1) // 2,), dtype=np.uint8)
            # Remove the edge between the vertices $n - 2$ and $n - 1$.
            flattened_colors[-1] = 0

            flattened_binary = np.expand_dims(flattened_colors, axis=0)

        super().__init__(
            bitmask_out=bitmask,
            bitmask_in=bitmask,
            adjacency_matrix_colors=adjacency_matrix_colors,
            adjacency_matrix_binary=adjacency_matrix_binary,
            flattened_row_major_colors=flattened_colors,
            flattened_row_major_binary=flattened_binary,
            flattened_clockwise_colors=flattened_colors,
            flattened_clockwise_binary=flattened_binary,
        )


# Although a complete bipartite graph is actually a complete $k$-partite graph for the case
# $k = 2$, this class directly inherits from the `Graph` class instead of the
# `CompleteKPartiteGraph` class for implementational efficiency and clarity.
class CompleteBipartiteGraph(Graph):
    r"""
    This class inherits from the `Graph` class and it is used to initialize complete bipartite
    graphs. In other words, it produces a 2-edge-colored loopless complete undirected graph such
    that its edges colored with the color 1 form a complete bipartite graph, while all the other
    edges are colored with the color 0. Here, the vertices $0, 1, 2, \\ldots, a_1 - 1$ form the
    first bipartition set, while the vertices $a_1, a_1 + 1, \\ldots, a_1 + a_2$ form the second
    bipartition set, where $a_1$ and $a_2$ represent the sizes of the two bipartition sets. It is
    also possible to configure which starting graph formats an instance should get initialized in,
    with the four accepted formats being the two bitmask formats (which are the same) and the two
    adjacency matrix formats.
    """

    def __init__(
        self, graph_formats: Set[GraphFormat], partition_size_1: int, partition_size_2: int
    ):
        """
        This constructor initializes the desired complete bipartite graph in the selected graph
        formats.

        :param graph_formats: A nonempty set containing items of the `GraphFormat` enumeration
            whose elements determine the starting graph formats that the graph should get
            initialized in. This set must contain at least one bitmask format or adjacency matrix
            format, while the flattened formats in the set are completely ignored.
        :param partition_size_1: A nonnegative `int` that represents $a_1$, i.e., the size of the
            first bipartition set.
        :param partition_size_2: A nonnegative `int` that represents $a_2$, i.e., the size of the
            second bipartition set. The graph order, i.e., the sum of the arguments
            ``partition_size_1`` and ``partition_size_2``, must be at least 1.
        """

        order = partition_size_1 + partition_size_2
        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None

        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            bitmask = np.full((1, order), (1 << partition_size_1) - 1, dtype=np.uint64)
            bitmask[0, :partition_size_1] = (1 << (order)) - (1 << partition_size_1)

        if graph_formats & {
            GraphFormat.ADJACENCY_MATRIX_COLORS,
            GraphFormat.ADJACENCY_MATRIX_BINARY,
        }:
            adjacency_matrix_colors = np.zeros((order, order), dtype=np.uint8)
            adjacency_matrix_colors[:partition_size_1, partition_size_1:] = 1
            adjacency_matrix_colors[partition_size_1:, :partition_size_1] = 1

            adjacency_matrix_binary = np.expand_dims(adjacency_matrix_colors, axis=0)

        super().__init__(
            bitmask_out=bitmask,
            bitmask_in=bitmask,
            adjacency_matrix_colors=adjacency_matrix_colors,
            adjacency_matrix_binary=adjacency_matrix_binary,
        )


class CompleteKPartiteGraph(Graph):
    r"""
    This class inherits from the `Graph` class and it is used to initialize complete $k$-partite
    graphs. In other words, it produces a 2-edge-colored loopless complete undirected graph such
    that its edges colored with the color 1 form a complete $k$-partite graph, while all the other
    edges are colored with the color 0. Here, the first $a_1$ vertices form the first $k$-partition
    set, the subsequent $a_2$ vertices form the second $k$-partition set, etc., and the last $a_k$
    vertices form the $k$-th $k$-partition set, with the parameters $k$ and $a_1, a_2, \\ldots,
    a_k$ being configurable. It is also possible to configure which starting graph formats an
    instance should get initialized in, with the four accepted formats being the two bitmask
    formats (which are the same) and the two adjacency matrix formats.
    """

    def __init__(self, graph_formats: Set[GraphFormat], partition_sizes: List[int]):
        r"""
        This constructor initializes the desired complete $k$-partite graph in the selected graph
        formats.

        :param graph_formats: A nonempty set containing items of the `GraphFormat` enumeration
            whose elements determine the starting graph formats that the graph should get
            initialized in. This set must contain at least one bitmask format or adjacency matrix
            format, while the flattened formats in the set are completely ignored.
        :param partition_sizes: A nonempty list of nonnegative `int` elements that are equal to
            $a_1, a_2, \\ldots, a_k$, respectively, with $k$ being positive. These elements
            represent the sizes of the partition sets and their sum must be at least 1.
        """

        order = sum(partition_sizes)
        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None

        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            # First, make all the vertices adjacent to all the other vertices, including loops.
            bitmask = np.full((1, order), (1 << order) - 1, dtype=np.uint64)

            # Then, iterate over the $k$-partition sets, and for each set, remove the edges between
            # these vertices, including loops.
            start = 0
            for item in partition_sizes:
                bitmask[0, start : start + item] -= (1 << (start + item)) - (1 << start)
                start += item

        if graph_formats & {
            GraphFormat.ADJACENCY_MATRIX_COLORS,
            GraphFormat.ADJACENCY_MATRIX_BINARY,
        }:
            # The same idea is used to construct the adjacency matrix. First make all the vertices
            # adjacent to each other, including loops, and then remove the unnecessary edges.
            adjacency_matrix_colors = np.ones((order, order), dtype=np.uint8)

            start = 0
            for item in partition_sizes:
                adjacency_matrix_colors[start : start + item, start : start + item] = 0
                start += item

            adjacency_matrix_binary = np.expand_dims(adjacency_matrix_colors, axis=0)

        super().__init__(
            bitmask_out=bitmask,
            bitmask_in=bitmask,
            adjacency_matrix_colors=adjacency_matrix_colors,
            adjacency_matrix_binary=adjacency_matrix_binary,
        )


class StarGraph(Graph):
    """
    This class inherits from the `Graph` class and it is used to instantiate star graphs. In other
    words, it produces a 2-edge-colored loopless complete undirected graph such that its edges
    colored with the color 1 form a star graph, while all the other edges are colored with the
    color 0. Recall that a star graph is a tree such that there is a vertex, called the central
    vertex, which is adjacent to all the other vertices. It is possible to select the central
    vertex and configure which starting graph formats an instance should get initialized in.
    """

    def __init__(self, graph_formats: Set[GraphFormat], graph_order: int, central_vertex: int = 0):
        """
        This constructor initializes the desired star graph in the selected graph formats.

        :param graph_formats: A nonempty set containing items of the `GraphFormat` enumeration
            whose elements determine the starting graph formats that the graph should get
            initialized in.
        :param graph_order: The graph order, given as a positive `int`.
        :param central_vertex: The central vertex of the star graph, given as a nonnegative `int`
            between 0 and ``graph_order - 1``. The default value is 0.
        """

        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None
        flattened_row_major_colors = None
        flattened_row_major_binary = None
        flattened_clockwise_colors = None
        flattened_clockwise_binary = None

        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            bitmask = np.full((1, graph_order), 1 << central_vertex, dtype=np.uint64)
            bitmask[0, central_vertex] = (1 << graph_order) - (1 << central_vertex) - 1

        if graph_formats & {
            GraphFormat.ADJACENCY_MATRIX_COLORS,
            GraphFormat.ADJACENCY_MATRIX_BINARY,
        }:
            adjacency_matrix_colors = np.zeros((graph_order, graph_order), dtype=np.uint8)
            adjacency_matrix_colors[central_vertex, :] = 1
            adjacency_matrix_colors[:, central_vertex] = 1
            adjacency_matrix_colors[central_vertex, central_vertex] = 0

            adjacency_matrix_binary = np.expand_dims(adjacency_matrix_colors, axis=0)

        if graph_formats & {
            GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
            GraphFormat.FLATTENED_ROW_MAJOR_BINARY,
        }:
            flattened_row_major_colors = np.zeros(
                (graph_order * (graph_order - 1) // 2,), dtype=np.uint8
            )
            # Add the edges between the central vertex and the vertices that follow it.
            start = central_vertex * (2 * graph_order - 1 - central_vertex) // 2
            flattened_row_major_colors[start : start + graph_order - central_vertex - 1] = 1

            # Add the edges between the central vertex and the vertices that precede it.
            indices = np.arange(central_vertex, dtype=np.int16)
            indices = indices * (2 * graph_order - 3 - indices) // 2 + central_vertex - 1
            flattened_row_major_colors[indices] = 1

            flattened_row_major_binary = np.expand_dims(flattened_row_major_colors, axis=0)

        if graph_formats & {
            GraphFormat.FLATTENED_CLOCKWISE_COLORS,
            GraphFormat.FLATTENED_CLOCKWISE_BINARY,
        }:
            flattened_clockwise_colors = np.zeros(
                (graph_order * (graph_order - 1) // 2,), dtype=np.uint8
            )
            # Add the edges between the central vertex and the vertices that precede it.
            start = central_vertex * (central_vertex - 1) // 2
            flattened_clockwise_colors[start : start + central_vertex] = 1

            # Add the edges between the central vertex and the vertices that follow it.
            indices = np.arange(central_vertex + 1, graph_order, dtype=np.int16)
            indices = indices * (indices - 1) // 2 + central_vertex
            flattened_clockwise_colors[indices] = 1

            flattened_clockwise_binary = np.expand_dims(flattened_clockwise_colors, axis=0)

        super().__init__(
            bitmask_out=bitmask,
            bitmask_in=bitmask,
            adjacency_matrix_colors=adjacency_matrix_colors,
            adjacency_matrix_binary=adjacency_matrix_binary,
            flattened_row_major_colors=flattened_row_major_colors,
            flattened_row_major_binary=flattened_row_major_binary,
            flattened_clockwise_colors=flattened_clockwise_colors,
            flattened_clockwise_binary=flattened_clockwise_binary,
        )


class PathGraph(Graph):
    r"""
    This class inherits from the `Graph` class and it is used to instantiate path graphs. In other
    words, it produces a 2-edge-colored loopless complete undirected graph such that its edges
    colored with the color 1 form a path graph, while all the other edges are colored with the
    color 0. In the said path graph, the vertices are $0, 1, 2, \\ldots, n - 1$, where $n$ is the
    graph order, with two vertices being adjacent if and only if they represent consecutive
    integers. It is also possible to configure which starting graph formats an instance should get
    initialized in.

    :note: The two bitmask formats can only be used if the path graph order is at most 63.
    """

    def __init__(self, graph_formats: Set[GraphFormat], graph_order: int):
        """
        This constructor initializes the desired path graph in the selected graph formats.

        :param graph_formats: A nonempty set containing items of the `GraphFormat` enumeration
            whose elements determine the starting graph formats that the graph should get
            initialized in.
        :param graph_order: The graph order, given as a positive `int`.
        """

        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None
        flattened_row_major_colors = None
        flattened_row_major_binary = None
        flattened_clockwise_colors = None
        flattened_clockwise_binary = None

        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            if graph_order >= 2:
                bitmask = ((1 << np.arange(graph_order, dtype=np.uint64)) * 5 // 2).reshape(1, -1)
                bitmask[0, 0] = 2
                bitmask[0, -1] = 1 << (graph_order - 2)
            # The trivial path needs to be settled separately.
            else:
                bitmask = np.zeros((1, 1), dtype=np.uint64)

        if graph_formats & {
            GraphFormat.ADJACENCY_MATRIX_COLORS,
            GraphFormat.ADJACENCY_MATRIX_BINARY,
        }:
            adjacency_matrix_colors = np.zeros((graph_order, graph_order), dtype=np.uint8)
            rows = np.arange(graph_order - 1, dtype=np.uint8)
            adjacency_matrix_colors[rows, rows + 1] = 1
            adjacency_matrix_colors[rows + 1, rows] = 1

            adjacency_matrix_binary = np.expand_dims(adjacency_matrix_colors, axis=0)

        if graph_formats & {
            GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
            GraphFormat.FLATTENED_ROW_MAJOR_BINARY,
        }:
            flattened_row_major_colors = np.zeros(
                (graph_order * (graph_order - 1) // 2,), dtype=np.uint8
            )
            indices = np.arange(graph_order - 1, dtype=np.int16)
            indices = indices * (2 * graph_order - 1 - indices) // 2
            flattened_row_major_colors[indices] = 1

            flattened_row_major_binary = np.expand_dims(flattened_row_major_colors, axis=0)

        if graph_formats & {
            GraphFormat.FLATTENED_CLOCKWISE_COLORS,
            GraphFormat.FLATTENED_CLOCKWISE_BINARY,
        }:
            flattened_clockwise_colors = np.zeros(
                (graph_order * (graph_order - 1) // 2,), dtype=np.uint8
            )
            indices = np.arange(2, graph_order + 1, dtype=np.int16)
            indices = indices * (indices - 1) // 2 - 1
            flattened_clockwise_colors[indices] = 1

            flattened_clockwise_binary = np.expand_dims(flattened_clockwise_colors, axis=0)

        super().__init__(
            bitmask_out=bitmask,
            bitmask_in=bitmask,
            adjacency_matrix_colors=adjacency_matrix_colors,
            adjacency_matrix_binary=adjacency_matrix_binary,
            flattened_row_major_colors=flattened_row_major_colors,
            flattened_row_major_binary=flattened_row_major_binary,
            flattened_clockwise_colors=flattened_clockwise_colors,
            flattened_clockwise_binary=flattened_clockwise_binary,
        )


class CycleGraph(Graph):
    r"""
    This class inherits from the `Graph` class and it is used to instantiate cycle graphs. In other
    words, it produces a 2-edge-colored loopless complete undirected graph such that its edges
    colored with the color 1 form a cycle graph, while all the other edges are colored with the
    color 0. In the said cycle graph, the vertices are $0, 1, 2, \\ldots, n - 1$, where $n$ is the
    graph order, with two vertices being adjacent if and only if they represent consecutive
    integers or they are 0 and $n - 1$. The positive integer $n$ must be at least 3. It is also
    possible to configure which starting graph formats an instance should get initialized in.

    :note: The two bitmask formats can only be used if the cycle graph order is at most 63.
    """

    def __init__(self, graph_formats: Set[GraphFormat], graph_order: int):
        """
        This constructor initializes the desired cycle graph in the selected graph formats.

        :param graph_formats: A nonempty set containing items of the `GraphFormat` enumeration
            whose elements determine the starting graph formats that the graph should get
            initialized in.
        :param graph_order: The graph order, given as a positive `int` not below 3.
        """

        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None
        flattened_row_major_colors = None
        flattened_row_major_binary = None
        flattened_clockwise_colors = None
        flattened_clockwise_binary = None

        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            bitmask = ((1 << np.arange(graph_order, dtype=np.uint64)) * 5 // 2).reshape(1, -1)
            bitmask[0, 0] = (1 << (graph_order - 1)) + 2
            bitmask[0, -1] = (1 << (graph_order - 2)) + 1

        if graph_formats & {
            GraphFormat.ADJACENCY_MATRIX_COLORS,
            GraphFormat.ADJACENCY_MATRIX_BINARY,
        }:
            adjacency_matrix_colors = np.zeros((graph_order, graph_order), dtype=np.uint8)
            # Add the edges from the path $0, 1, 2, \ldots, n - 1$.
            rows = np.arange(graph_order - 1, dtype=np.uint8)
            adjacency_matrix_colors[rows, rows + 1] = 1
            adjacency_matrix_colors[rows + 1, rows] = 1
            # Add the edge between the vertices 0 and $n - 1$.
            adjacency_matrix_colors[0, -1] = 1
            adjacency_matrix_colors[-1, 0] = 1

            adjacency_matrix_binary = np.expand_dims(adjacency_matrix_colors, axis=0)

        if graph_formats & {
            GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
            GraphFormat.FLATTENED_ROW_MAJOR_BINARY,
        }:
            flattened_row_major_colors = np.zeros(
                (graph_order * (graph_order - 1) // 2,), dtype=np.uint8
            )
            # Add the edges from the path $0, 1, 2, \ldots, n - 1$.
            indices = np.arange(graph_order - 1, dtype=np.int16)
            indices = indices * (2 * graph_order - 1 - indices) // 2
            flattened_row_major_colors[indices] = 1
            # Add the edge between the vertices 0 and $n - 1$.
            flattened_row_major_colors[graph_order - 2] = 1

            flattened_row_major_binary = np.expand_dims(flattened_row_major_colors, axis=0)

        if graph_formats & {
            GraphFormat.FLATTENED_CLOCKWISE_COLORS,
            GraphFormat.FLATTENED_CLOCKWISE_BINARY,
        }:
            flattened_clockwise_colors = np.zeros(
                (graph_order * (graph_order - 1) // 2,), dtype=np.uint8
            )
            # Add the edges from the path $0, 1, 2, \ldots, n - 1$.
            indices = np.arange(2, graph_order + 1, dtype=np.int16)
            indices = indices * (indices - 1) // 2 - 1
            flattened_clockwise_colors[indices] = 1
            # Add the edge between the vertices 0 and $n - 1$.
            flattened_clockwise_colors[(graph_order - 1) * (graph_order - 2) // 2] = 1

            flattened_clockwise_binary = np.expand_dims(flattened_clockwise_colors, axis=0)

        super().__init__(
            bitmask_out=bitmask,
            bitmask_in=bitmask,
            adjacency_matrix_colors=adjacency_matrix_colors,
            adjacency_matrix_binary=adjacency_matrix_binary,
            flattened_row_major_colors=flattened_row_major_colors,
            flattened_row_major_binary=flattened_row_major_binary,
            flattened_clockwise_colors=flattened_clockwise_colors,
            flattened_clockwise_binary=flattened_clockwise_binary,
        )


class WheelGraph(Graph):
    r"""
    This class inherits from the `Graph` class and it is used to instantiate wheel graphs. In other
    words, it produces a 2-edge-colored loopless complete undirected graph such that its edges
    colored with the color 1 form a wheel graph, while all the other edges are colored with the
    color 0. In the said wheel graph, the vertices are $0, 1, 2, \\ldots, n - 1$, where $n$ is the
    graph order, and vertex 0 is adjacent to all the other vertices, while the subgraph induced by
    the remaining vertices forms the cycle $1, 2, 3, \\ldots, n - 1, 1$. The positive integer $n$
    must be at least 4. It is also possible to configure which starting graph formats an instance
    should get initialized in.

    :note: The two bitmask formats can only be used if the wheel graph order is at most 63.
    """

    def __init__(self, graph_formats: Set[GraphFormat], graph_order: int):
        """
        This constructor initializes the desired wheel graph in the selected graph formats.

        :param graph_formats: A nonempty set containing items of the `GraphFormat` enumeration
            whose elements determine the starting graph formats that the graph should get
            initialized in.
        :param graph_order: The graph order, given as a positive `int` not below 4.
        """

        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None
        flattened_row_major_colors = None
        flattened_row_major_binary = None
        flattened_clockwise_colors = None
        flattened_clockwise_binary = None

        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            bitmask = ((1 << np.arange(graph_order, dtype=np.uint64)) * 5 // 2 + 1).reshape(1, -1)
            bitmask[0, 0] = (1 << graph_order) - 2
            bitmask[0, 1] = (1 << (graph_order - 1)) + 5
            bitmask[0, -1] = (1 << (graph_order - 2)) + 3

        if graph_formats & {
            GraphFormat.ADJACENCY_MATRIX_COLORS,
            GraphFormat.ADJACENCY_MATRIX_BINARY,
        }:
            adjacency_matrix_colors = np.zeros((graph_order, graph_order), dtype=np.uint8)
            # Add the edges from 0 to all the other vertices.
            adjacency_matrix_colors[0, 1:] = 1
            adjacency_matrix_colors[1:, 0] = 1
            # Add the edges from the path $1, 2, 3, \ldots, n - 1$.
            rows = np.arange(1, graph_order - 1, dtype=np.uint8)
            adjacency_matrix_colors[rows, rows + 1] = 1
            adjacency_matrix_colors[rows + 1, rows] = 1
            # Add the edge between the vertices 1 and $n - 1$.
            adjacency_matrix_colors[1, -1] = 1
            adjacency_matrix_colors[-1, 1] = 1

            adjacency_matrix_binary = np.expand_dims(adjacency_matrix_colors, axis=0)

        if graph_formats & {
            GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
            GraphFormat.FLATTENED_ROW_MAJOR_BINARY,
        }:
            flattened_row_major_colors = np.zeros(
                (graph_order * (graph_order - 1) // 2,), dtype=np.uint8
            )
            # Add the edges from 0 to all the other vertices.
            flattened_row_major_colors[: graph_order - 1] = 1
            # Add the edges from the path $1, 2, 3, \ldots, n - 1$.
            indices = np.arange(1, graph_order - 1, dtype=np.int16)
            indices = indices * (2 * graph_order - 1 - indices) // 2
            flattened_row_major_colors[indices] = 1
            # Add the edge between the vertices 1 and $n - 1$.
            flattened_row_major_colors[2 * graph_order - 4] = 1

            flattened_row_major_binary = np.expand_dims(flattened_row_major_colors, axis=0)

        if graph_formats & {
            GraphFormat.FLATTENED_CLOCKWISE_COLORS,
            GraphFormat.FLATTENED_CLOCKWISE_BINARY,
        }:
            flattened_clockwise_colors = np.zeros(
                (graph_order * (graph_order - 1) // 2,), dtype=np.uint8
            )
            # Add the edges from 0 to all the other vertices.
            indices = np.arange(1, graph_order, dtype=np.int16)
            indices = indices * (indices - 1) // 2
            flattened_clockwise_colors[indices] = 1
            # Add the edges from the path $1, 2, 3, \ldots, n - 1$.
            indices = np.arange(3, graph_order + 1, dtype=np.int16)
            indices = indices * (indices - 1) // 2 - 1
            flattened_clockwise_colors[indices] = 1
            # Add the edge between the vertices 1 and $n - 1$.
            flattened_clockwise_colors[(graph_order - 1) * (graph_order - 2) // 2 + 1] = 1

            flattened_clockwise_binary = np.expand_dims(flattened_clockwise_colors, axis=0)

        super().__init__(
            bitmask_out=bitmask,
            bitmask_in=bitmask,
            adjacency_matrix_colors=adjacency_matrix_colors,
            adjacency_matrix_binary=adjacency_matrix_binary,
            flattened_row_major_colors=flattened_row_major_colors,
            flattened_row_major_binary=flattened_row_major_binary,
            flattened_clockwise_colors=flattened_clockwise_colors,
            flattened_clockwise_binary=flattened_clockwise_binary,
        )


class BookGraph(Graph):
    r"""
    This class inherits from the `Graph` class and it is used to instantiate book graphs. In other
    words, it produces a 2-edge-colored loopless complete undirected graph such that its edges
    colored with the color 1 form a book graph, while all the other edges are colored with the
    color 0. In the said book graph, the vertices are $0, 1, 2, \\ldots, m + 1$, where $m$ is the
    book graph index, and vertices 0 and 1 are adjacent to all the other vertices, while the
    remaining $m$ vertices are not adjacent to one another. It is also possible to configure which
    starting graph formats an instance should get initialized in.
    """

    def __init__(self, graph_formats: Set[GraphFormat], index: int):
        """
        This constructor initializes the desired book graph in the selected graph formats.

        :param graph_formats: A nonempty set containing items of the `GraphFormat` enumeration
            whose elements determine the starting graph formats that the graph should get
            initialized in.
        :param index: The book graph index $m$, given as a positive `int`. Note that the graph
            order is equal to ``index + 2``.
        """

        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None
        flattened_row_major_colors = None
        flattened_row_major_binary = None
        flattened_clockwise_colors = None
        flattened_clockwise_binary = None

        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            bitmask = np.full((1, index + 2), 3, dtype=np.uint64)
            bitmask[0, 0] = (1 << (index + 2)) - 2
            bitmask[0, 1] = (1 << (index + 2)) - 3

        if graph_formats & {
            GraphFormat.ADJACENCY_MATRIX_COLORS,
            GraphFormat.ADJACENCY_MATRIX_BINARY,
        }:
            adjacency_matrix_colors = np.zeros((index + 2, index + 2), dtype=np.uint8)
            # Add all the possible edges with one endpoint from \{ 0, 1 \} and the other from
            # \{ 2, 3, \ldots, m + 1 \}.
            adjacency_matrix_colors[:2, 2:] = 1
            adjacency_matrix_colors[2:, :2] = 1
            # Add the edge between the vertices 0 and 1.
            adjacency_matrix_colors[0, 1] = 1
            adjacency_matrix_colors[1, 0] = 1

            adjacency_matrix_binary = np.expand_dims(adjacency_matrix_colors, axis=0)

        if graph_formats & {
            GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
            GraphFormat.FLATTENED_ROW_MAJOR_BINARY,
        }:
            flattened_row_major_colors = np.zeros(
                ((index + 2) * (index + 1) // 2,), dtype=np.uint8
            )
            flattened_row_major_colors[: 2 * index + 1] = 1

            flattened_row_major_binary = np.expand_dims(flattened_row_major_colors, axis=0)

        if graph_formats & {
            GraphFormat.FLATTENED_CLOCKWISE_COLORS,
            GraphFormat.FLATTENED_CLOCKWISE_BINARY,
        }:
            flattened_clockwise_colors = np.zeros(
                ((index + 2) * (index + 1) // 2,), dtype=np.uint8
            )
            # Add the edge between the vertices 0 and 1.
            flattened_clockwise_colors[0] = 1

            indices = np.arange(2, index + 2, dtype=np.int16)
            indices = indices * (indices - 1) // 2
            # Add all the possible edges with one endpoint from \{ 0, 1 \} and the other from
            # \{ 2, 3, \ldots, m + 1 \}.
            flattened_clockwise_colors[indices] = 1
            flattened_clockwise_colors[indices + 1] = 1

            flattened_clockwise_binary = np.expand_dims(flattened_clockwise_colors, axis=0)

        super().__init__(
            bitmask_out=bitmask,
            bitmask_in=bitmask,
            adjacency_matrix_colors=adjacency_matrix_colors,
            adjacency_matrix_binary=adjacency_matrix_binary,
            flattened_row_major_colors=flattened_row_major_colors,
            flattened_row_major_binary=flattened_row_major_binary,
            flattened_clockwise_colors=flattened_clockwise_colors,
            flattened_clockwise_binary=flattened_clockwise_binary,
        )


class FriendshipGraph(Graph):
    r"""
    This class inherits from the `Graph` class and it is used to instantiate friendship graphs. In
    other words, it produces a 2-edge-colored loopless complete undirected graph such that its
    edges colored with the color 1 form a friendship graph, while all the other edges are colored
    with the color 0. In the said friendship graph, the vertices are $0, 1, 2, \\ldots, 2m - 1,
    2m$, where $m$ is the friendship graph index, and vertex 0 is adjacent to all the other
    vertices, while the remaining $2m$ vertices have exactly one neighbor among themselves,
    determined in the following manner:

    * if $i$ is odd, then vertex $i$ is adjacent only to $i + 1$; and
    * if $i$ is even, then vertex $i$ is adjacent only to $i - 1$.

    It is also possible to configure which starting graph formats an instance should get
    initialized in.
    """

    def __init__(self, graph_formats: Set[GraphFormat], index: int):
        """
        This constructor initializes the desired friendship graph in the selected graph formats.

        :param graph_formats: A nonempty set containing items of the `GraphFormat` enumeration
            whose elements determine the starting graph formats that the graph should get
            initialized in.
        :param index: The friendship graph index $m$, given as a positive `int`. Note that the
            graph order is equal to ``2 * index + 1``.
        """

        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None
        flattened_row_major_colors = None
        flattened_row_major_binary = None
        flattened_clockwise_colors = None
        flattened_clockwise_binary = None

        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            bitmask = np.full((1, 2 * index + 1), 1, dtype=np.uint64)
            bitmask[0, 0] = (1 << (2 * index + 1)) - 2
            bitmask[0, 1::2] += 1 << np.arange(2, 2 * index + 2, 2, dtype=np.uint64)
            bitmask[0, 2::2] += 1 << np.arange(1, 2 * index + 1, 2, dtype=np.uint64)

        if graph_formats & {
            GraphFormat.ADJACENCY_MATRIX_COLORS,
            GraphFormat.ADJACENCY_MATRIX_BINARY,
        }:
            adjacency_matrix_colors = np.zeros((2 * index + 1, 2 * index + 1), dtype=np.uint8)
            # Add the edges from 0 to all the other vertices.
            adjacency_matrix_colors[0, 1:] = 1
            adjacency_matrix_colors[1:, 0] = 1
            # Add the remaining edges, i.e., the edges of the form $\{ i, i + 1 \}$, where $i \in
            # \{ 1, 3, 5, 7, \ldots, 2m - 1 \}$.
            rows = np.arange(1, 2 * index + 1, 2, dtype=np.uint8)
            adjacency_matrix_colors[rows, rows + 1] = 1
            adjacency_matrix_colors[rows + 1, rows] = 1

            adjacency_matrix_binary = np.expand_dims(adjacency_matrix_colors, axis=0)

        if graph_formats & {
            GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
            GraphFormat.FLATTENED_ROW_MAJOR_BINARY,
        }:
            flattened_row_major_colors = np.zeros(((2 * index + 1) * index,), dtype=np.uint8)
            # Add the edges from 0 to all the other vertices.
            flattened_row_major_colors[: 2 * index] = 1
            # Add the remaining edges, i.e., the edges of the form $\{ i, i + 1 \}$, where $i \in
            # \{ 1, 3, 5, 7, \ldots, 2m - 1 \}$.
            indices = np.arange(1, 2 * index + 1, 2, dtype=np.int16)
            indices = indices * (4 * index + 1 - indices) // 2
            flattened_row_major_colors[indices] = 1

            flattened_row_major_binary = np.expand_dims(flattened_row_major_colors, axis=0)

        if graph_formats & {
            GraphFormat.FLATTENED_CLOCKWISE_COLORS,
            GraphFormat.FLATTENED_CLOCKWISE_BINARY,
        }:
            flattened_clockwise_colors = np.zeros(((2 * index + 1) * index,), dtype=np.uint8)
            # Add the edges from 0 to all the other vertices.
            indices = np.arange(1, 2 * index + 1, dtype=np.int16)
            indices = indices * (indices - 1) // 2
            flattened_clockwise_colors[indices] = 1
            # Add the remaining edges, i.e., the edges of the form $\{ i, i + 1 \}$, where $i \in
            # \{ 1, 3, 5, 7, \ldots, 2m - 1 \}$.
            indices = np.arange(3, 2 * index + 3, 2, dtype=np.int16)
            indices = indices * (indices - 1) // 2 - 1
            flattened_clockwise_colors[indices] = 1

            flattened_clockwise_binary = np.expand_dims(flattened_clockwise_colors, axis=0)

        super().__init__(
            bitmask_out=bitmask,
            bitmask_in=bitmask,
            adjacency_matrix_colors=adjacency_matrix_colors,
            adjacency_matrix_binary=adjacency_matrix_binary,
            flattened_row_major_colors=flattened_row_major_colors,
            flattened_row_major_binary=flattened_row_major_binary,
            flattened_clockwise_colors=flattened_clockwise_colors,
            flattened_clockwise_binary=flattened_clockwise_binary,
        )
