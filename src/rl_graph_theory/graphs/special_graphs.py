"""
This ``Python`` module contains various classes that inherit the `Graph` class and are used to
construct edge-colored complete graphs with some particular structure.
"""

from typing import List

import numpy as np

from .graph import Graph, GraphFormat


class MonochromaticGraph(Graph):
    """
    This class inherits the `Graph` class and it is used to instantiate monochromatic graphs, i.e.,
    graphs where all the edges are colored in the same color (or they are all uncolored). It is
    also possible to configure the starting graph format that an instance should get initialized
    in.
    """

    def __init__(
        self, graph_format: GraphFormat, order: int, edge_colors: int, selected_edge_color: int
    ):
        """
        This constructor initializes the desired monochromatic graph in the selected graph format.

        :param graph_format: The starting graph format that the graph should get initialized in,
            given as an item of the `GraphFormat` enumeration.
        :param order: The graph order, given as a positive integer.
        :param edge_colors: A positive integer (not below two) that represents the number of
            available edge colors.
        :param selected_edge_color: The edge color that all the edges should get colored in, given
            as a nonnegative integer between 0 and ``edge_colors``. If this argument equals
            ``edge_colors``, then this means that all the edges should be uncolored.
        """

        bitmask = None
        adjacency_matrix = None
        flattened_column_first = None
        flattened_row_first = None

        if graph_format == GraphFormat.BITMASK:
            if selected_edge_color == edge_colors:
                # If the graph order is at least two, then it contains edges, hence there are
                # uncolored edges.
                if order >= 2:
                    bitmask = np.zeros((edge_colors, order), dtype=int)
                # Otherwise, the graph contains no edges, which means that there are no uncolored
                # edges.
                else:
                    bitmask = np.zeros((edge_colors - 1, order), dtype=int)
            else:
                bitmask = np.zeros((edge_colors - 1, order), dtype=int)
                if selected_edge_color != 0:
                    bitmask[selected_edge_color - 1, :] = (1 << order) - 1
                    bitmask[selected_edge_color - 1, :] -= 1 << np.arange(order, dtype=int)

        elif graph_format == GraphFormat.ADJACENCY_MATRIX:
            adjacency_matrix = np.full((order, order), selected_edge_color, dtype=int)
            if selected_edge_color != 0:
                np.fill_diagonal(adjacency_matrix, 0)

        elif graph_format == GraphFormat.FLATTENED_CLOCKWISE:
            flattened_column_first = np.full(
                (order * (order - 1) // 2,), selected_edge_color, dtype=int
            )

        else:
            flattened_row_first = np.full(
                (order * (order - 1) // 2,), selected_edge_color, dtype=int
            )

        super().__init__(
            graph_format=graph_format,
            edge_colors=edge_colors,
            bitmask=bitmask,
            adjacency_matrix=adjacency_matrix,
            flattened_column_major=flattened_column_first,
            flattened_row_major=flattened_row_first,
        )


# Although an empty graph can conceptually be regarded as a monochromatic graph, this class
# directly inherits the `Graph` class instead of the `MonochromaticGraph` class for
# implementational reasons.
class EmptyGraph(Graph):
    """
    This class inherits the `Graph` class and it is used to instantiate empty graphs. In other
    words, it produces a graph of a given order and with two available edge colors such that all of
    its edges are colored in the color 0. It is also possible to configure the starting graph
    format that an instance should get initialized in.
    """

    def __init__(self, graph_format: GraphFormat, order: int):
        """
        This constructor initializes the desired empty graph in the selected graph format.

        :param graph_format: The starting graph format that the graph should get initialized in,
            given as an item of the `GraphFormat` enumeration.
        :param order: The graph order, given as a positive integer.
        """

        bitmask = None
        adjacency_matrix = None
        flattened_column_first = None
        flattened_row_first = None

        if graph_format == GraphFormat.BITMASK:
            bitmask = np.zeros((1, order), dtype=int)
        elif graph_format == GraphFormat.ADJACENCY_MATRIX:
            adjacency_matrix = np.zeros((order, order), dtype=int)
        elif graph_format == GraphFormat.FLATTENED_CLOCKWISE:
            flattened_column_first = np.zeros((order * (order - 1) // 2,), dtype=int)
        else:
            flattened_row_first = np.zeros((order * (order - 1) // 2,), dtype=int)

        super().__init__(
            graph_format=graph_format,
            edge_colors=2,
            bitmask=bitmask,
            adjacency_matrix=adjacency_matrix,
            flattened_column_major=flattened_column_first,
            flattened_row_major=flattened_row_first,
        )


# Although a complete graph can conceptually be regarded as a monochromatic graph, this class
# directly inherits the `Graph` class instead of the `MonochromaticGraph` class for
# implementational reasons.
class CompleteGraph(Graph):
    """
    This class inherits the `Graph` class and it is used to instantiate complete graphs. In other
    words, it produces a graph of a given order and with two available edge colors such that all of
    its edges are colored in the color 1. It is also possible to configure the starting graph
    format that an instance should get initialized in.
    """

    def __init__(self, graph_format: GraphFormat, order: int):
        """
        This constructor initializes the desired complete graph in the selected graph format.

        :param graph_format: The starting graph format that the graph should get initialized in,
            given as an item of the `GraphFormat` enumeration.
        :param order: The graph order, given as a positive integer.
        """

        bitmask = None
        adjacency_matrix = None
        flattened_column_first = None
        flattened_row_first = None

        if graph_format == GraphFormat.BITMASK:
            bitmask = np.full((1, order), (1 << order) - 1, dtype=int)
            bitmask[0, :] -= 1 << np.arange(order, dtype=int)
        elif graph_format == GraphFormat.ADJACENCY_MATRIX:
            adjacency_matrix = np.ones((order, order), dtype=int)
            np.fill_diagonal(adjacency_matrix, 0)
        elif graph_format == GraphFormat.FLATTENED_CLOCKWISE:
            flattened_column_first = np.ones((order * (order - 1) // 2,), dtype=int)
        else:
            flattened_row_first = np.ones((order * (order - 1) // 2,), dtype=int)

        super().__init__(
            graph_format=graph_format,
            edge_colors=2,
            bitmask=bitmask,
            adjacency_matrix=adjacency_matrix,
            flattened_column_major=flattened_column_first,
            flattened_row_major=flattened_row_first,
        )


class AlmostCompleteGraph(Graph):
    r"""
    This class inherits the `Graph` class and it is used to instantiate almost-complete graphs. In
    other words, it produces a graph of a given order $n$ and with the vertex set $\\{ 0, 1, 2,
    \\ldots, n - 1 \\}$ such that it has two available edge colors and all of its edges are colored
    in the color 1, except for the edge between the vertices $n - 2$ and $n - 1$, which is colored
    in the color 0. It is also possible to configure the starting graph format that an instance
    should get initialized in.
    """

    def __init__(self, graph_format: GraphFormat, order: int):
        """
        This constructor initializes the desired almost-complete graph in the selected graph
        format.

        :param graph_format: The starting graph format the graph should get initialized in, given
            as an item of the `GraphFormat` enumeration.
        :param order: The graph order $n$, given as a positive integer (not below two).
        """

        bitmask = None
        adjacency_matrix = None
        flattened_column_first = None
        flattened_row_first = None

        if graph_format == GraphFormat.BITMASK:
            bitmask = np.full((1, order), (1 << order) - 1, dtype=int)
            bitmask[0, :] -= 1 << np.arange(order, dtype=int)
            # Remove the edge between the vertices $n - 2$ and $n - 1$.
            bitmask[0, -1] -= 1 << (order - 2)
            bitmask[0, -2] -= 1 << (order - 1)

        elif graph_format == GraphFormat.ADJACENCY_MATRIX:
            adjacency_matrix = np.ones((order, order), dtype=int)
            np.fill_diagonal(adjacency_matrix, 0)
            # Remove the edge between the vertices $n - 2$ and $n - 1$.
            adjacency_matrix[-1, -2] = 0
            adjacency_matrix[-2, -1] = 0

        elif graph_format == GraphFormat.FLATTENED_CLOCKWISE:
            flattened_column_first = np.ones((order * (order - 1) // 2,), dtype=int)
            # Remove the edge between the vertices $n - 2$ and $n - 1$.
            flattened_column_first[-1] = 0

        else:
            flattened_row_first = np.ones((order * (order - 1) // 2,), dtype=int)
            # Remove the edge between the vertices $n - 2$ and $n - 1$.
            flattened_row_first[-1] = 0

        super().__init__(
            graph_format=graph_format,
            edge_colors=2,
            bitmask=bitmask,
            adjacency_matrix=adjacency_matrix,
            flattened_column_major=flattened_column_first,
            flattened_row_major=flattened_row_first,
        )


# Although a complete bipartite graph is actually a complete $k$-partite graph for the case
# $k = 2$, this class directly inherits the `Graph` class instead of the `CompleteKPartiteGraph`
# class for implementational efficiency and clarity.
class CompleteBipartiteGraph(Graph):
    r"""
    This class inherits the `Graph` class and it is used to initialize complete bipartite graphs.
    In other words, it produces a graph with two available edge colors such that its edges colored
    in the color 1 form a complete bipartite graph, while all the other edges are colored in the
    color 0. Here, the vertices $0, 1, 2, \\ldots, a_1 - 1$ form the first bipartition set, while
    the vertices $a_1, a_1 + 1, \\ldots, a_1 + a_2$ form the second bipartition set, where $a_1$
    and $a_2$ represent the sizes of the two bipartition sets. It is also possible to configure the
    starting graph format that an instance should get initialized in, with the two possible formats
    being the bitmask format and the adjacency matrix format.
    """

    def __init__(self, graph_format: GraphFormat, partition_size_1: int, partition_size_2: int):
        """
        This constructor initializes the desired complete bipartite graph in the selected graph
        format.

        :param graph_format: The starting graph format the graph should get initialized in, given
            as one of the two items `GraphFormat.BITMASK` and `GraphFormat.ADJACENCY_MATRIX` from
            the `GraphFormat` enumeration.
        :param partition_size_1: A nonnegative integer that represents $a_1$, i.e., the size of the
            first bipartition set.
        :param partition_size_2: A nonnegative integer that represents $a_2$, i.e., the size of the
            second bipartition set. The graph order, i.e., the sum of the arguments
            ``partition_size_1`` and ``partition_size_2``, must be at least one.
        """

        order = partition_size_1 + partition_size_2
        bitmask = None
        adjacency_matrix = None

        if graph_format == GraphFormat.BITMASK:
            bitmask = np.full((1, order), (1 << partition_size_1) - 1, dtype=int)
            bitmask[0, :partition_size_1] = (1 << (order)) - (1 << partition_size_1)

        elif graph_format == GraphFormat.ADJACENCY_MATRIX:
            adjacency_matrix = np.zeros((order, order), dtype=int)
            adjacency_matrix[:partition_size_1, partition_size_1:] = 1
            adjacency_matrix[partition_size_1:, :partition_size_1] = 1

        else:
            raise ValueError

        super().__init__(
            graph_format=graph_format,
            edge_colors=2,
            bitmask=bitmask,
            adjacency_matrix=adjacency_matrix,
        )


class CompleteKPartiteGraph(Graph):
    r"""
    This class inherits the `Graph` class and it is used to initialize complete $k$-partite graphs.
    In other words, it produces a graph with two available edge colors such that its edges colored
    in the color 1 form a complete $k$-partite graph, while all the other edges are colored in the
    color 0. Here, the first $a_1$ vertices form the first $k$-partition set, the subsequent $a_2$
    vertices form the second $k$-partition set, etc., and the last $a_k$ vertices form the $k$-th
    $k$-partition set, with the parameters $k$ and $a_1, a_2, \\ldots, a_k$ being configurable. It
    is also possible to configure the starting graph format that an instance should get initialized
    in, with the two possible formats being the bitmask format and the adjacency matrix format.
    """

    def __init__(self, graph_format: GraphFormat, partition_sizes: List[int]):
        r"""
        This constructor initializes the desired complete $k$-partite graph in the selected graph
        format.

        :param graph_format: The starting graph format the graph should get initialized in, given
            as one of the two items `GraphFormat.BITMASK` and `GraphFormat.ADJACENCY_MATRIX` from
            the `GraphFormat` enumeration.
        :param partition_sizes: A nonempty list of nonnegative integers whose elements are equal to
            $a_1, a_2, \\ldots, a_k$, respectively, with $k$ being positive. These elements
            represent the sizes of the partition sets and their sum must be at least one.
        """

        order = sum(partition_sizes)
        bitmask = None
        adjacency_matrix = None

        if graph_format == GraphFormat.BITMASK:
            # First, make all the vertices adjacent to all the other vertices, including loops.
            bitmask = np.full((1, order), (1 << order) - 1, dtype=int)

            # Then, iterate over the $k$-partition sets, and for each set, remove the edges between
            # these vertices, including loops.
            start = 0
            for item in partition_sizes:
                bitmask[0, start : start + item] -= (1 << (start + item)) - (1 << start)
                start += item

        elif graph_format == GraphFormat.ADJACENCY_MATRIX:
            # The same idea is used to construct the adjacency matrix. First make all the vertices
            # adjacent to each other, including loops, and then remove the unnecessary edges.
            adjacency_matrix = np.ones((order, order), dtype=int)

            start = 0
            for item in partition_sizes:
                adjacency_matrix[start : start + item, start : start + item] = 0
                start += item

        else:
            raise ValueError

        super().__init__(
            graph_format=graph_format,
            edge_colors=2,
            bitmask=bitmask,
            adjacency_matrix=adjacency_matrix,
        )


class StarGraph(Graph):
    """
    This class inherits the `Graph` class and it is used to instantiate star graphs. In other
    words, it produces a graph of a given order and with two available edge colors such that its
    edges colored in the color 1 form a star graph, while all the other edges are colored in the
    color 0. Recall that a star graph is a tree such that there is a vertex, called the central
    vertex, which is adjacent to all the other vertices. It is possible to select the central
    vertex and configure the starting graph format that an instance should get initialized in.
    """

    def __init__(self, graph_format: GraphFormat, order: int, central_vertex: int = 0):
        """
        This constructor initializes the desired star graph in the selected graph format.

        :param graph_format: The starting graph format that the graph should get initialized in,
            given as an item of the `GraphFormat` enumeration.
        :param order: The graph order, given as a positive integer.
        :param central_vertex: The central vertex of the star graph, given as a nonnegative integer
            between 0 and ``order - 1``. The default value is 0.
        """

        bitmask = None
        adjacency_matrix = None
        flattened_column_first = None
        flattened_row_first = None

        if graph_format == GraphFormat.BITMASK:
            bitmask = np.full((1, order), 1 << central_vertex, dtype=int)
            bitmask[0, central_vertex] = (1 << order) - (1 << central_vertex) - 1

        elif graph_format == GraphFormat.ADJACENCY_MATRIX:
            adjacency_matrix = np.zeros((order, order), dtype=int)
            adjacency_matrix[central_vertex, :] = 1
            adjacency_matrix[:, central_vertex] = 1
            adjacency_matrix[central_vertex, central_vertex] = 0

        elif graph_format == GraphFormat.FLATTENED_CLOCKWISE:
            flattened_column_first = np.zeros((order * (order - 1) // 2,), dtype=int)
            # Add the edges between the central vertex and the vertices that precede it.
            start = central_vertex * (central_vertex - 1) // 2
            flattened_column_first[start : start + central_vertex] = 1

            # Add the edges between the central vertex and the vertices that follow it.
            indices = np.arange(central_vertex + 1, order, dtype=int)
            indices = indices * (indices - 1) // 2 + central_vertex
            flattened_column_first[indices] = 1

        else:
            flattened_row_first = np.zeros((order * (order - 1) // 2,), dtype=int)
            # Add the edges between the central vertex and the vertices that follow it.
            start = central_vertex * (2 * order - 1 - central_vertex) // 2
            flattened_row_first[start : start + order - central_vertex - 1] = 1

            # Add the edges between the central vertex and the vertices that precede it.
            indices = np.arange(central_vertex, dtype=int)
            indices = indices * (2 * order - 3 - indices) // 2 + central_vertex - 1
            flattened_row_first[indices] = 1

        super().__init__(
            graph_format=graph_format,
            edge_colors=2,
            bitmask=bitmask,
            adjacency_matrix=adjacency_matrix,
            flattened_column_major=flattened_column_first,
            flattened_row_major=flattened_row_first,
        )


class PathGraph(Graph):
    r"""
    This class inherits the `Graph` class and it is used to instantiate path graphs. In other
    words, it produces a graph of a given order and with two available edge colors such that its
    edges colored in the color 1 form a path graph, while all the other edges are colored in the
    color 0. In the said path graph, the vertices are $0, 1, 2, \\ldots, n - 1$, where $n$ is the
    graph order, with two vertices being adjacent if and only if they represent consecutive
    integers. It is also possible to configure the starting graph format that an instance should
    get initialized in.
    """

    def __init__(self, graph_format: GraphFormat, order: int):
        """
        This constructor initializes the desired path graph in the selected graph format.

        :param graph_format: The starting graph format that the graph should get initialized in,
            given as an item of the `GraphFormat` enumeration.
        :param order: The graph order $n$, given as a positive integer.
        """

        bitmask = None
        adjacency_matrix = None
        flattened_column_first = None
        flattened_row_first = None

        if graph_format == GraphFormat.BITMASK:
            if order >= 2:
                bitmask = ((1 << np.arange(order, dtype=int)) * 5 // 2).reshape(1, -1)
                bitmask[0, 0] = 2
                bitmask[0, -1] = 1 << (order - 2)
            # The trivial path needs to be settled separately.
            else:
                bitmask = np.zeros((1, 1), dtype=int)

        elif graph_format == GraphFormat.ADJACENCY_MATRIX:
            adjacency_matrix = np.zeros((order, order), dtype=int)
            rows = np.arange(order - 1, dtype=int)
            adjacency_matrix[rows, rows + 1] = 1
            adjacency_matrix[rows + 1, rows] = 1

        elif graph_format == GraphFormat.FLATTENED_CLOCKWISE:
            flattened_column_first = np.zeros((order * (order - 1) // 2,), dtype=int)
            indices = np.arange(2, order + 1, dtype=int)
            indices = indices * (indices - 1) // 2 - 1
            flattened_column_first[indices] = 1

        else:
            flattened_row_first = np.zeros((order * (order - 1) // 2,), dtype=int)
            indices = np.arange(order - 1, dtype=int)
            indices = indices * (2 * order - 1 - indices) // 2
            flattened_row_first[indices] = 1

        super().__init__(
            graph_format=graph_format,
            edge_colors=2,
            bitmask=bitmask,
            adjacency_matrix=adjacency_matrix,
            flattened_column_major=flattened_column_first,
            flattened_row_major=flattened_row_first,
        )


class CycleGraph(Graph):
    r"""
    This class inherits the `Graph` class and it is used to instantiate cycle graphs. In other
    words, it produces a graph of a given order and with two available edge colors such that its
    edges colored in the color 1 form a cycle graph, while all the other edges are colored in the
    color 0. In the said cycle graph, the vertices are $0, 1, 2, \\ldots, n - 1$, where $n$ is the
    graph order, with two vertices being adjacent if and only if they represent consecutive
    integers or they are 0 and $n - 1$. The positive integer $n$ must be at least three. It is also
    possible to configure the starting graph format that an instance should get initialized in.
    """

    def __init__(self, graph_format: GraphFormat, order: int):
        """
        This constructor initializes the desired cycle graph in the selected graph format.

        :param graph_format: The starting graph format that the graph should get initialized in,
            given as an item of the `GraphFormat` enumeration.
        :param order: A positive integer (not below three) that represents the graph order $n$.
        """

        bitmask = None
        adjacency_matrix = None
        flattened_column_first = None
        flattened_row_first = None

        if graph_format == GraphFormat.BITMASK:
            bitmask = ((1 << np.arange(order, dtype=int)) * 5 // 2).reshape(1, -1)
            bitmask[0, 0] = (1 << (order - 1)) + 2
            bitmask[0, -1] = (1 << (order - 2)) + 1

        elif graph_format == GraphFormat.ADJACENCY_MATRIX:
            adjacency_matrix = np.zeros((order, order), dtype=int)
            # Add the edges from the path $0, 1, 2, \ldots, n - 1$.
            rows = np.arange(order - 1, dtype=int)
            adjacency_matrix[rows, rows + 1] = 1
            adjacency_matrix[rows + 1, rows] = 1
            # Add the edge between the vertices 0 and $n - 1$.
            adjacency_matrix[0, -1] = 1
            adjacency_matrix[-1, 0] = 1

        elif graph_format == GraphFormat.FLATTENED_CLOCKWISE:
            flattened_column_first = np.zeros((order * (order - 1) // 2,), dtype=int)
            # Add the edges from the path $0, 1, 2, \ldots, n - 1$.
            indices = np.arange(2, order + 1, dtype=int)
            indices = indices * (indices - 1) // 2 - 1
            flattened_column_first[indices] = 1
            # Add the edge between the vertices 0 and $n - 1$.
            flattened_column_first[(order - 1) * (order - 2) // 2] = 1

        else:
            flattened_row_first = np.zeros((order * (order - 1) // 2,), dtype=int)
            # Add the edges from the path $0, 1, 2, \ldots, n - 1$.
            indices = np.arange(order - 1, dtype=int)
            indices = indices * (2 * order - 1 - indices) // 2
            flattened_row_first[indices] = 1
            # Add the edge between the vertices 0 and $n - 1$.
            flattened_row_first[order - 2] = 1

        super().__init__(
            graph_format=graph_format,
            edge_colors=2,
            bitmask=bitmask,
            adjacency_matrix=adjacency_matrix,
            flattened_column_major=flattened_column_first,
            flattened_row_major=flattened_row_first,
        )


class WheelGraph(Graph):
    r"""
    This class inherits the `Graph` class and it is used to instantiate wheel graphs. In other
    words, it produces a graph of a given order and with two available edge colors such that its
    edges colored in the color 1 form a wheel graph, while all the other edges are colored in the
    color 0. In the said wheel graph, the vertices are $0, 1, 2, \\ldots, n - 1$, where $n$ is the
    graph order, and vertex 0 is adjacent to all the other vertices, while the subgraph induced by
    the remaining vertices forms the cycle $1, 2, 3, \\ldots, n - 1, 1$. The positive integer $n$
    must be at least four. It is also possible to configure the starting graph format that an
    instance should get initialized in.
    """

    def __init__(self, graph_format: GraphFormat, order: int):
        """
        This constructor initializes the desired wheel graph in the selected graph format.

        :param graph_format: The starting graph format that the graph should get initialized in,
            given as an item of the `GraphFormat` enumeration.
        :param order: A positive integer (not below four) that represents the graph order $n$.
        """

        bitmask = None
        adjacency_matrix = None
        flattened_column_first = None
        flattened_row_first = None

        if graph_format == GraphFormat.BITMASK:
            bitmask = ((1 << np.arange(order, dtype=int)) * 5 // 2 + 1).reshape(1, -1)
            bitmask[0, 0] = (1 << order) - 2
            bitmask[0, 1] = (1 << (order - 1)) + 5
            bitmask[0, -1] = (1 << (order - 2)) + 3

        elif graph_format == GraphFormat.ADJACENCY_MATRIX:
            adjacency_matrix = np.zeros((order, order), dtype=int)
            # Add the edges from 0 to all the other vertices.
            adjacency_matrix[0, 1:] = 1
            adjacency_matrix[1:, 0] = 1
            # Add the edges from the path $1, 2, 3, \ldots, n - 1$.
            rows = np.arange(1, order - 1, dtype=int)
            adjacency_matrix[rows, rows + 1] = 1
            adjacency_matrix[rows + 1, rows] = 1
            # Add the edge between the vertices 1 and $n - 1$.
            adjacency_matrix[1, -1] = 1
            adjacency_matrix[-1, 1] = 1

        elif graph_format == GraphFormat.FLATTENED_CLOCKWISE:
            flattened_column_first = np.zeros((order * (order - 1) // 2,), dtype=int)
            # Add the edges from 0 to all the other vertices.
            indices = np.arange(1, order, dtype=int)
            indices = indices * (indices - 1) // 2
            flattened_column_first[indices] = 1
            # Add the edges from the path $1, 2, 3, \ldots, n - 1$.
            indices = np.arange(3, order + 1, dtype=int)
            indices = indices * (indices - 1) // 2 - 1
            flattened_column_first[indices] = 1
            # Add the edge between the vertices 1 and $n - 1$.
            flattened_column_first[(order - 1) * (order - 2) // 2 + 1] = 1

        else:
            flattened_row_first = np.zeros((order * (order - 1) // 2,), dtype=int)
            # Add the edges from 0 to all the other vertices.
            flattened_row_first[: order - 1] = 1
            # Add the edges from the path $1, 2, 3, \ldots, n - 1$.
            indices = np.arange(1, order - 1, dtype=int)
            indices = indices * (2 * order - 1 - indices) // 2
            flattened_row_first[indices] = 1
            # Add the edge between the vertices 1 and $n - 1$.
            flattened_row_first[2 * order - 4] = 1

        super().__init__(
            graph_format=graph_format,
            edge_colors=2,
            bitmask=bitmask,
            adjacency_matrix=adjacency_matrix,
            flattened_column_major=flattened_column_first,
            flattened_row_major=flattened_row_first,
        )


class BookGraph(Graph):
    r"""
    This class inherits the `Graph` class and it is used to instantiate book graphs. In other
    words, it produces a graph with two available edge colors such that its edges colored in the
    color 1 form a book graph, while all the other edges are colored in the color 0. In the said
    book graph, the vertices are $0, 1, 2, \\ldots, n + 1$, where $n$ is the book graph index, and
    vertices 0 and 1 are adjacent to all the other vertices, while the remaining $n$ vertices are
    not adjacent to one another. It is also possible to configure the starting graph format that an
    instance should get initialized in.
    """

    def __init__(self, graph_format: GraphFormat, index: int):
        """
        This constructor initializes the desired book graph in the selected graph format.

        :param graph_format: The starting graph format that the graph should get initialized in,
            given as an item of the `GraphFormat` enumeration.
        :param index: The book graph index $n$, given as a positive integer. Note that the graph
            order is equal to ``index + 2``.
        """

        bitmask = None
        adjacency_matrix = None
        flattened_column_first = None
        flattened_row_first = None

        if graph_format == GraphFormat.BITMASK:
            bitmask = np.full((1, index + 2), 3, dtype=int)
            bitmask[0, 0] = (1 << (index + 2)) - 2
            bitmask[0, 1] = (1 << (index + 2)) - 3

        elif graph_format == GraphFormat.ADJACENCY_MATRIX:
            adjacency_matrix = np.zeros((index + 2, index + 2), dtype=int)
            # Add all the possible edges with one endpoint from \{ 0, 1 \} and the other from
            # \{ 2, 3, \ldots, n + 1 \}.
            adjacency_matrix[:2, 2:] = 1
            adjacency_matrix[2:, :2] = 1
            # Add the edge between the vertices 0 and 1.
            adjacency_matrix[0, 1] = 1
            adjacency_matrix[1, 0] = 1

        elif graph_format == GraphFormat.FLATTENED_CLOCKWISE:
            flattened_column_first = np.zeros(((index + 2) * (index + 1) // 2,), dtype=int)
            # Add the edge between the vertices 0 and 1.
            flattened_column_first[0] = 1

            indices = np.arange(2, index + 2, dtype=int)
            indices = indices * (indices - 1) // 2
            # Add all the possible edges with one endpoint from \{ 0, 1 \} and the other from
            # \{ 2, 3, \ldots, n + 1 \}.
            flattened_column_first[indices] = 1
            flattened_column_first[indices + 1] = 1

        else:
            flattened_row_first = np.zeros(((index + 2) * (index + 1) // 2,), dtype=int)
            flattened_row_first[: 2 * index + 1] = 1

        super().__init__(
            graph_format=graph_format,
            edge_colors=2,
            bitmask=bitmask,
            adjacency_matrix=adjacency_matrix,
            flattened_column_major=flattened_column_first,
            flattened_row_major=flattened_row_first,
        )


class FriendshipGraph(Graph):
    r"""
    This class inherits the `Graph` class and it is used to instantiate friendship graphs. In other
    words, it produces a graph with two available edge colors such that its edges colored in the
    color 1 form a friendship graph, while all the other edges are colored in the color 0. In the
    said friendship graph, the vertices are $0, 1, 2, \\ldots, 2n - 1, 2n$, where $n$ is the
    friendship graph index, and vertex 0 is adjacent to all the other vertices, while the remaining
    $2n$ vertices have exactly one neighbor among themselves, determined in the following manner:

    * if $i$ is odd, then vertex $i$ is adjacent only to $i + 1$; and
    * if $i$ is even, then vertex $i$ is adjacent only to $i - 1$.

    It is also possible to configure the starting graph format that an instance should get
    initialized in.
    """

    def __init__(self, graph_format: GraphFormat, index: int):
        """
        This constructor initializes the desired friendship graph in the selected graph format.

        :param graph_format: The starting graph format that the graph should get initialized in,
            given as an item of the `GraphFormat` enumeration.
        :param index: The friendship graph index $n$, given as a positive integer. Note that the
            graph order is equal to ``2 * index + 1``.
        """

        bitmask = None
        adjacency_matrix = None
        flattened_column_first = None
        flattened_row_first = None

        if graph_format == GraphFormat.BITMASK:
            bitmask = np.full((1, 2 * index + 1), 1, dtype=int)
            bitmask[0, 0] = (1 << (2 * index + 1)) - 2
            bitmask[0, 1::2] += 1 << np.arange(2, 2 * index + 2, 2, dtype=int)
            bitmask[0, 2::2] += 1 << np.arange(1, 2 * index + 1, 2, dtype=int)

        elif graph_format == GraphFormat.ADJACENCY_MATRIX:
            adjacency_matrix = np.zeros((2 * index + 1, 2 * index + 1), dtype=int)
            # Add the edges from 0 to all the other vertices.
            adjacency_matrix[0, 1:] = 1
            adjacency_matrix[1:, 0] = 1
            # Add the remaining edges, i.e., the edges of the form $\{ i, i + 1 \}$, where $i \in
            # \{ 1, 3, 5, 7, \ldots, 2n - 1 \}$.
            rows = np.arange(1, 2 * index + 1, 2)
            adjacency_matrix[rows, rows + 1] = 1
            adjacency_matrix[rows + 1, rows] = 1

        elif graph_format == GraphFormat.FLATTENED_CLOCKWISE:
            flattened_column_first = np.zeros(((2 * index + 1) * index,), dtype=int)
            # Add the edges from 0 to all the other vertices.
            indices = np.arange(1, 2 * index + 1, dtype=int)
            indices = indices * (indices - 1) // 2
            flattened_column_first[indices] = 1
            # Add the remaining edges, i.e., the edges of the form $\{ i, i + 1 \}$, where $i \in
            # \{ 1, 3, 5, 7, \ldots, 2n - 1 \}$.
            indices = np.arange(3, 2 * index + 3, 2, dtype=int)
            indices = indices * (indices - 1) // 2 - 1
            flattened_column_first[indices] = 1

        else:
            flattened_row_first = np.zeros(((2 * index + 1) * index,), dtype=int)
            # Add the edges from 0 to all the other vertices.
            flattened_row_first[: 2 * index] = 1
            # Add the remaining edges, i.e., the edges of the form $\{ i, i + 1 \}$, where $i \in
            # \{ 1, 3, 5, 7, \ldots, 2n - 1 \}$.
            indices = np.arange(1, 2 * index + 1, 2, dtype=int)
            indices = indices * (4 * index + 1 - indices) // 2
            flattened_row_first[indices] = 1

        super().__init__(
            graph_format=graph_format,
            edge_colors=2,
            bitmask=bitmask,
            adjacency_matrix=adjacency_matrix,
            flattened_column_major=flattened_column_first,
            flattened_row_major=flattened_row_first,
        )
