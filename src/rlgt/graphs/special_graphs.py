"""
This ``Python`` module contains classes that inherit from the `Graph` class and provide ways
to construct $k$-edge-colored looped complete graphs with specific structures.
"""

from typing import List, Set

import numpy as np

from .graph import Graph
from .graph_formats import GraphFormat
from .utils import graph_order_to_flattened_length


class MonochromaticGraph(Graph):
    """
    This class inherits from the `Graph` class and is used to create monochromatic graphs, i.e.,
    $k$-edge-colored looped complete graphs where all edges (resp. arcs) share the same color or
    are all uncolored. The class also allows selecting the initial graph formats in which an
    instance should be represented.
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
        This constructor initializes a monochromatic graph in the specified graph formats.

        :param graph_formats: A nonempty set of `GraphFormat` enumeration items that specify the
            formats in which the graph should be initialized.
        :param graph_order: The graph order, given as a positive `int`.
        :param edge_colors: A positive `int` (at least 2) representing the number of proper edge
            colors, i.e., $k$. The default value is 2.
        :param selected_color: The color to assign to all edges (resp. arcs), given as a
            nonnegative `int` from 0 to ``edge_colors``. If equal to ``edge_colors``, all edges
            (resp. arcs) are uncolored. The default value is 0.
        :param is_directed: A `bool` that indicates whether the graph is directed (`True`) or
            undirected (`False`). The default value is `False`.
        :param allow_loops: A `bool` that indicates whether loops are allowed in the graph. The
            default is `False`.
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
        # Use the reduced formats if the graph is fully colored.
        if not_fully_colored:
            color_dim = edge_colors
        else:
            color_dim = edge_colors - 1

        # Compute the flattened length for the flattened formats.
        flattened_length = graph_order_to_flattened_length(
            graph_order=graph_order,
            is_directed=is_directed,
            allow_loops=allow_loops,
        )

        # Initialize the bitmask formats if requested.
        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            bitmask = np.zeros((color_dim, graph_order), dtype=np.uint64)

            if selected_color != 0 and selected_color != edge_colors:
                bitmask[selected_color - 1, :] = (1 << graph_order) - 1
                # Remove the loops if they are not allowed.
                if not allow_loops:
                    bitmask[selected_color - 1, :] -= 1 << np.arange(graph_order, dtype=np.uint64)

        # Initialize the adjacency matrix format with color numbers if requested.
        if GraphFormat.ADJACENCY_MATRIX_COLORS in graph_formats:
            adjacency_matrix_colors = np.full(
                (graph_order, graph_order), selected_color, dtype=np.uint8
            )
            # Remove the loops if they are not allowed.
            if not allow_loops and selected_color != 0:
                np.fill_diagonal(adjacency_matrix_colors, 0)

        # Initialize the adjacency matrix format with binary slices if requested.
        if GraphFormat.ADJACENCY_MATRIX_BINARY in graph_formats:
            adjacency_matrix_binary = np.zeros(
                (color_dim, graph_order, graph_order), dtype=np.uint8
            )

            if selected_color != 0 and selected_color != edge_colors:
                adjacency_matrix_binary[selected_color] = 1
                # Remove the loops if they are not allowed.
                if not allow_loops:
                    np.fill_diagonal(adjacency_matrix_binary[selected_color], 0)

        # Initialize the flattened formats with color numbers if requested.
        if graph_formats & {
            GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
            GraphFormat.FLATTENED_CLOCKWISE_COLORS,
        }:
            flattened_colors = np.full((flattened_length), selected_color, dtype=np.uint8)

        # Initialize the flattened formats with binary slices if requested.
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
# inherits directly from the `Graph` class rather than `MonochromaticGraph` for implementation
# reasons.
class EmptyGraph(Graph):
    """
    This class inherits from the `Graph` class and is used to instantiate empty graphs.
    Specifically, it produces a 2-edge-colored loopless complete undirected graph of a given order
    in which all edges are colored with color 0. The class also allows selecting the initial graph
    formats in which an instance should be represented.
    """

    def __init__(self, graph_formats: Set[GraphFormat], graph_order: int):
        """
        This constructor initializes the empty graph in the selected graph formats.

        :param graph_formats: A nonempty set of `GraphFormat` enumeration items that specify the
            formats in which the graph should be initialized.
        :param graph_order: The graph order, given as a positive `int`.
        """

        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None
        flattened_colors = None
        flattened_binary = None

        # Initialize the bitmask formats if requested.
        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            bitmask = np.zeros((1, graph_order), dtype=np.uint64)

        # Initialize the adjacency matrix formats if requested.
        if graph_formats & {
            GraphFormat.ADJACENCY_MATRIX_COLORS,
            GraphFormat.ADJACENCY_MATRIX_BINARY,
        }:
            adjacency_matrix_colors = np.zeros((graph_order, graph_order), dtype=np.uint8)
            adjacency_matrix_binary = np.expand_dims(adjacency_matrix_colors, axis=0)

        # Initialize the flattened formats if requested.
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
# inherits directly from the `Graph` class rather than `MonochromaticGraph` for implementation
# reasons.
class CompleteGraph(Graph):
    """
    This class inherits from the `Graph` class and is used to instantiate complete graphs.
    Specifically, it produces a 2-edge-colored loopless complete undirected graph of a given order
    in which all edges are colored with color 1. The class also allows selecting the initial graph
    formats in which an instance should be represented.
    """

    def __init__(self, graph_formats: Set[GraphFormat], graph_order: int):
        """
        This constructor initializes the complete graph in the selected graph formats.

        :param graph_formats: A nonempty set of `GraphFormat` enumeration items that specify the
            formats in which the graph should be initialized.
        :param graph_order: The graph order, given as a positive `int`.
        """

        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None
        flattened_colors = None
        flattened_binary = None

        # Initialize the bitmask formats if requested.
        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            bitmask = np.full((1, graph_order), (1 << graph_order) - 1, dtype=np.uint64)
            bitmask[0, :] -= 1 << np.arange(graph_order, dtype=np.uint64)

        # Initialize the adjacency matrix formats if requested.
        if graph_formats & {
            GraphFormat.ADJACENCY_MATRIX_COLORS,
            GraphFormat.ADJACENCY_MATRIX_BINARY,
        }:
            adjacency_matrix_colors = np.ones((graph_order, graph_order), dtype=np.uint8)
            np.fill_diagonal(adjacency_matrix_colors, 0)
            adjacency_matrix_binary = np.expand_dims(adjacency_matrix_colors, axis=0)

        # Initialize the flattened formats if requested.
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
    This class inherits from the `Graph` class and is used to instantiate almost-complete graphs.
    Specifically, it produces a 2-edge-colored loopless complete undirected graph of order $n$ in
    which all edges are colored with color 1, except for the edge between vertices $n - 2$ and
    $n - 1$, which is colored with color 0. The class also allows selecting the initial graph
    formats in which an instance should be represented.
    """

    def __init__(self, graph_formats: Set[GraphFormat], graph_order: int):
        """
        This constructor initializes the almost-complete graph in the selected graph formats.

        :param graph_formats: A nonempty set of `GraphFormat` enumeration items that specify the
            formats in which the graph should be initialized.
        :param graph_order: The graph order, given as a positive `int`.
        """

        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None
        flattened_colors = None
        flattened_binary = None

        # Initialize the bitmask formats if requested.
        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            bitmask = np.full((1, graph_order), (1 << graph_order) - 1, dtype=np.uint64)
            bitmask[0, :] -= 1 << np.arange(graph_order, dtype=np.uint64)
            # Remove the edge between the vertices $n - 2$ and $n - 1$.
            bitmask[0, -1] -= 1 << (graph_order - 2)
            bitmask[0, -2] -= 1 << (graph_order - 1)

        # Initialize the adjacency matrix formats if requested.
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

        # Initialize the flattened formats if requested.
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


# Although a complete bipartite graph is a special case of a complete $k$-partite graph with
# $k = 2$, this class directly inherits from the `Graph` class instead of the
# `CompleteKPartiteGraph` class for reasons of implementation efficiency and clarity.
class CompleteBipartiteGraph(Graph):
    r"""
    This class inherits from the `Graph` class and is used to instantiate complete
    bipartite graphs. Specifically, it produces a 2-edge-colored loopless complete undirected graph
    in which the edges colored with color 1 form a complete bipartite graph, while all remaining
    edges are colored with color 0. The vertex set is partitioned into two parts. The vertices $0,
    1, 2, \\ldots, a_1 - 1$ form the first bipartition set, and the vertices $a_1, a_1 + 1,
    \\ldots, a_1 + a_2 - 1$ form the second bipartition set, where $a_1$ and $a_2$ denote the sizes
    of the two bipartition sets. The instance can be initialized in selected graph formats. Only
    the bitmask formats and the adjacency matrix formats are supported; any flattened formats
    provided are ignored.
    """

    def __init__(
        self, graph_formats: Set[GraphFormat], partition_size_1: int, partition_size_2: int
    ):
        """
        This constructor initializes the complete bipartite graph in the selected graph formats.

        :param graph_formats: A nonempty set of `GraphFormat` items that determines which graph
            formats the graph should be initialized in. The set must contain at least one bitmask
            format or adjacency matrix format. Any flattened formats in the set are ignored.
        :param partition_size_1: A nonnegative `int` representing $a_1$, the size of the first
            bipartition set.
        :param partition_size_2: A nonnegative `int` representing $a_2$, the size of the second
            bipartition set. The graph order, equal to ``partition_size_1 + partition_size_2``,
            must be at least 1.
        """

        order = partition_size_1 + partition_size_2
        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None

        # Initialize the bitmask formats if requested.
        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            bitmask = np.full((1, order), (1 << partition_size_1) - 1, dtype=np.uint64)
            bitmask[0, :partition_size_1] = (1 << (order)) - (1 << partition_size_1)

        # Initialize the adjacency matrix formats if requested.
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
    This class inherits from the `Graph` class and is used to instantiate complete $k$-partite
    graphs. Specifically, it produces a 2-edge-colored loopless complete undirected graph in which
    the edges colored with color 1 form a complete $k$-partite graph, while all remaining edges are
    colored with color 0. The vertex set is partitioned into $k$ parts. The first $a_1$ vertices
    form the first partition set, the next $a_2$ vertices form the second partition set, and so on,
    with the final $a_k$ vertices forming the $k$-th partition set. The parameters $k$ and $a_1,
    a_2, \\ldots, a_k$ are determined by the list of partition sizes provided at construction time.
    The instance can be initialized in selected graph formats. Only the bitmask formats and the
    adjacency matrix formats are supported; any flattened formats provided are ignored.
    """

    def __init__(self, graph_formats: Set[GraphFormat], partition_sizes: List[int]):
        r"""
        This constructor initializes the complete $k$-partite graph in the selected graph formats.

        :param graph_formats: A nonempty set of `GraphFormat` items that determines which graph
            formats the graph should be initialized in. The set must contain at least one bitmask
            format or adjacency matrix format. Any flattened formats in the set are ignored.
        :param partition_sizes: A nonempty list of nonnegative `int` values $a_1, a_2, \\ldots,
            a_k$, where $k \ge 1$. These values specify the sizes of the partition sets, and their
            sum, which equals the graph order, must be at least 1.
        """

        order = sum(partition_sizes)
        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None

        # Initialize the bitmask formats if requested.
        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            # First, make all the vertices adjacent to all the other vertices, including loops.
            bitmask = np.full((1, order), (1 << order) - 1, dtype=np.uint64)

            # Then, iterate over the $k$-partition sets, and for each set, remove the edges between
            # these vertices, including loops.
            start = 0
            for item in partition_sizes:
                bitmask[0, start : start + item] -= (1 << (start + item)) - (1 << start)
                start += item

        # Initialize the adjacency matrix formats if requested.
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
    This class inherits from the `Graph` class and is used to instantiate star graphs.
    Specifically, it produces a 2-edge-colored loopless complete undirected graph in which the
    edges colored with color 1 form a star graph, while all remaining edges are colored with color
    0. A star graph is a tree in which there exists a vertex, called a central vertex, that is
    adjacent to every other vertex. The central vertex can be chosen explicitly at construction
    time. The class also allows selecting the initial graph formats in which an instance should be
    represented.
    """

    def __init__(self, graph_formats: Set[GraphFormat], graph_order: int, central_vertex: int = 0):
        """
        This constructor initializes the star graph in the selected graph formats.

        :param graph_formats: A nonempty set of `GraphFormat` items that determines which graph
            formats the graph should be initialized in.
        :param graph_order: The graph order, given as a positive `int`.
        :param central_vertex: The index of the central vertex, given as a nonnegative `int`
            between 0 and ``graph_order - 1``. The default value is 0.
        """

        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None
        flattened_row_major_colors = None
        flattened_row_major_binary = None
        flattened_clockwise_colors = None
        flattened_clockwise_binary = None

        # Initialize the bitmask formats if requested.
        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            bitmask = np.full((1, graph_order), 1 << central_vertex, dtype=np.uint64)
            bitmask[0, central_vertex] = (1 << graph_order) - (1 << central_vertex) - 1

        # Initialize the adjacency matrix formats if requested.
        if graph_formats & {
            GraphFormat.ADJACENCY_MATRIX_COLORS,
            GraphFormat.ADJACENCY_MATRIX_BINARY,
        }:
            adjacency_matrix_colors = np.zeros((graph_order, graph_order), dtype=np.uint8)
            adjacency_matrix_colors[central_vertex, :] = 1
            adjacency_matrix_colors[:, central_vertex] = 1
            adjacency_matrix_colors[central_vertex, central_vertex] = 0

            adjacency_matrix_binary = np.expand_dims(adjacency_matrix_colors, axis=0)

        # Initialize the flattened row-major formats if requested.
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

        # Initialize the flattened clockwise formats if requested.
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
    This class inherits from the `Graph` class and is used to instantiate path graphs. It produces
    a 2-edge-colored loopless complete undirected graph in which the edges colored with the color
    1 form a path graph, while all other edges are colored with the color 0. The vertices are
    labeled $0, 1, 2, \\ldots, n - 1$, where $n$ is the graph order, and two vertices are adjacent
    if and only if their labels differ by exactly one. The class also allows selecting the initial
    graph formats in which an instance should be represented.

    :note: The two bitmask formats can only be used when the graph order is at most 63.
    """

    def __init__(self, graph_formats: Set[GraphFormat], graph_order: int):
        """
        This constructor initializes the path graph in the selected graph formats.

        :param graph_formats: A nonempty set of `GraphFormat` items that determines which graph
            formats the graph should be initialized in.
        :param graph_order: The graph order, given as a positive `int`.
        """

        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None
        flattened_row_major_colors = None
        flattened_row_major_binary = None
        flattened_clockwise_colors = None
        flattened_clockwise_binary = None

        # Initialize the bitmask formats if requested.
        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            if graph_order >= 2:
                bitmask = ((1 << np.arange(graph_order, dtype=np.uint64)) * 5 // 2).reshape(1, -1)
                bitmask[0, 0] = 2
                bitmask[0, -1] = 1 << (graph_order - 2)
            # The trivial path needs to be settled separately.
            else:
                bitmask = np.zeros((1, 1), dtype=np.uint64)

        # Initialize the adjacency matrix formats if requested.
        if graph_formats & {
            GraphFormat.ADJACENCY_MATRIX_COLORS,
            GraphFormat.ADJACENCY_MATRIX_BINARY,
        }:
            adjacency_matrix_colors = np.zeros((graph_order, graph_order), dtype=np.uint8)
            rows = np.arange(graph_order - 1, dtype=np.uint8)
            adjacency_matrix_colors[rows, rows + 1] = 1
            adjacency_matrix_colors[rows + 1, rows] = 1

            adjacency_matrix_binary = np.expand_dims(adjacency_matrix_colors, axis=0)

        # Initialize the flattened row-major formats if requested.
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

        # Initialize the flattened clockwise formats if requested.
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
    This class inherits from the `Graph` class and is used to instantiate cycle graphs. It produces
    a 2-edge-colored loopless complete undirected graph in which the edges colored with the color
    1 form a cycle graph, while all other edges are colored with the color 0. The vertices are
    labeled $0, 1, 2, \ldots, n - 1$, where $n$ is the graph order, and two vertices are adjacent
    if and only if their labels differ by exactly one modulo $n$. The graph order $n$ must be at
    least 3. The class also allows selecting the initial graph formats in which an instance should
    be represented.

    :note: The two bitmask formats can only be used when the graph order is at most 63.
    """

    def __init__(self, graph_formats: Set[GraphFormat], graph_order: int):
        """
        This constructor initializes the cycle graph in the selected graph formats.

        :param graph_formats: A nonempty set of `GraphFormat` items that determines which graph
            formats the graph should be initialized in.
        :param graph_order: The graph order, given as a positive `int` not below 3.
        """

        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None
        flattened_row_major_colors = None
        flattened_row_major_binary = None
        flattened_clockwise_colors = None
        flattened_clockwise_binary = None

        # Initialize the bitmask formats if requested.
        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            bitmask = ((1 << np.arange(graph_order, dtype=np.uint64)) * 5 // 2).reshape(1, -1)
            bitmask[0, 0] = (1 << (graph_order - 1)) + 2
            bitmask[0, -1] = (1 << (graph_order - 2)) + 1

        # Initialize the adjacency matrix formats if requested.
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

        # Initialize the flattened row-major formats if requested.
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

        # Initialize the flattened clockwise formats if requested.
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
    This class inherits from the `Graph` class and is used to instantiate wheel graphs. It produces
    a 2-edge-colored loopless complete undirected graph in which the edges colored with the color 1
    form a wheel graph, while all other edges are colored with the color 0. The vertices are
    labeled $0, 1, 2, \ldots, n - 1$, where $n$ is the graph order. Vertex 0 is adjacent to every
    other vertex, while the subgraph induced by the remaining vertices forms the cycle $(1, 2, 3,
    \\ldots, n - 1, 1)$. The graph order $n$ must be at least 4. The class also allows selecting
    the initial graph formats in which an instance should be represented.

    :note: The two bitmask formats can only be used when the graph order is at most 63.
    """

    def __init__(self, graph_formats: Set[GraphFormat], graph_order: int):
        """
        This constructor initializes the wheel graph in the selected graph formats.

        :param graph_formats: A nonempty set of `GraphFormat` items that determines which graph
            formats the graph should be initialized in.
        :param graph_order: The graph order, given as a positive `int` not below 4.
        """

        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None
        flattened_row_major_colors = None
        flattened_row_major_binary = None
        flattened_clockwise_colors = None
        flattened_clockwise_binary = None

        # Initialize the bitmask formats if requested.
        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            bitmask = ((1 << np.arange(graph_order, dtype=np.uint64)) * 5 // 2 + 1).reshape(1, -1)
            bitmask[0, 0] = (1 << graph_order) - 2
            bitmask[0, 1] = (1 << (graph_order - 1)) + 5
            bitmask[0, -1] = (1 << (graph_order - 2)) + 3

        # Initialize the adjacency matrix formats if requested.
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

        # Initialize the flattened row-major formats if requested.
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

        # Initialize the flattened clockwise formats if requested.
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
    This class inherits from the `Graph` class and is used to instantiate book graphs.
    Specifically, it produces a 2-edge-colored loopless complete undirected graph such that its
    edges colored with the color 1 form a book graph, while all other edges are colored with the
    color 0. A book graph of index $m$ is a simple graph on $m + 2$ vertices where vertices 0 and 1
    are adjacent to each other and to all remaining vertices, and the remaining $m$ vertices are
    independent, i.e., no edges exist between them. The class also allows selecting the initial
    graph formats in which an instance should be represented.
    """

    def __init__(self, graph_formats: Set[GraphFormat], index: int):
        """
        This constructor initializes the book graph in the selected graph formats.

        :param graph_formats: A nonempty set of `GraphFormat` items that determines which graph
            formats the graph should be initialized in.
        :param index: The book graph index $m$, given as a positive `int`. The graph order is equal
            to ``index + 2``.
        """

        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None
        flattened_row_major_colors = None
        flattened_row_major_binary = None
        flattened_clockwise_colors = None
        flattened_clockwise_binary = None

        # Initialize the bitmask formats if requested.
        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            bitmask = np.full((1, index + 2), 3, dtype=np.uint64)
            bitmask[0, 0] = (1 << (index + 2)) - 2
            bitmask[0, 1] = (1 << (index + 2)) - 3

        # Initialize the adjacency matrix formats if requested.
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

        # Initialize the flattened row-major formats if requested.
        if graph_formats & {
            GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
            GraphFormat.FLATTENED_ROW_MAJOR_BINARY,
        }:
            flattened_row_major_colors = np.zeros(
                ((index + 2) * (index + 1) // 2,), dtype=np.uint8
            )
            flattened_row_major_colors[: 2 * index + 1] = 1

            flattened_row_major_binary = np.expand_dims(flattened_row_major_colors, axis=0)

        # Initialize the flattened clockwise formats if requested.
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
    This class inherits from the `Graph` class and is used to instantiate friendship graphs.
    Specifically, it produces a 2-edge-colored loopless complete undirected graph such that its
    edges colored with the color 1 form a friendship graph, while all other edges are colored with
    the color 0. A friendship graph of index $m$ is a simple graph on $2m + 1$ vertices where
    vertex 0 is adjacent to all remaining vertices, and the other $2m$ vertices form $m$
    independent edges: if a vertex $i$ is odd, it is adjacent only to $i + 1$, and if $i$ is even,
    it is adjacent only to $i - 1$. The class also allows selecting the initial graph formats in
    which an instance should be represented.
    """

    def __init__(self, graph_formats: Set[GraphFormat], index: int):
        """
        This constructor initializes the friendship graph in the selected graph formats.

        :param graph_formats: A nonempty set of `GraphFormat` items that determines which graph
            formats the graph should be initialized in.
        :param index: The friendship graph index $m$, given as a positive `int`. The graph order is
            equal to ``2 * index + 1``.
        """

        bitmask = None
        adjacency_matrix_colors = None
        adjacency_matrix_binary = None
        flattened_row_major_colors = None
        flattened_row_major_binary = None
        flattened_clockwise_colors = None
        flattened_clockwise_binary = None

        # Initialize the bitmask formats if requested.
        if graph_formats & {GraphFormat.BITMASK_OUT, GraphFormat.BITMASK_IN}:
            bitmask = np.full((1, 2 * index + 1), 1, dtype=np.uint64)
            bitmask[0, 0] = (1 << (2 * index + 1)) - 2
            bitmask[0, 1::2] += 1 << np.arange(2, 2 * index + 2, 2, dtype=np.uint64)
            bitmask[0, 2::2] += 1 << np.arange(1, 2 * index + 1, 2, dtype=np.uint64)

        # Initialize the adjacency matrix formats if requested.
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

        # Initialize the flattened row-major formats if requested.
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

        # Initialize the flattened clockwise formats if requested.
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
