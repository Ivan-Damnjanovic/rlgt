"""
This ``Python`` module contains various classes that inherit the `Graph` class and are used to
construct concrete edge-colored complete graphs with a given structure.
"""

import numpy as np
from typing import List

from .graph import Graph, GraphFormat, EdgeOrdering


class EmptyGraph(Graph):
    """
    This class inherits the `Graph` class and it is used to instantiate empty graphs. In other
    words, it produces a graph of a given order and with a given number of available edge colors
    such that all of its edges are colored in the color 0. It is also possible to configure the
    starting graph format that the instance should be initialized in.
    """

    def __init__(self, graph_format: GraphFormat, order: int, edge_colors: int = 2):
        """
        This constructor initializes the desired empty graph in the selected graph format.

        :param graph_format: The starting graph format the graph should be initialized in, given as
            an item of the `GraphFormat` enumeration.
        :param order: The graph order, given as a positive integer.
        :param edge_colors: A positive integer (not below two) that determines the number of
            available edge colors. The default value is 2.
        """

        bitmask = None
        adjacency_matrix = None
        flattened_column_first = None
        flattened_row_first = None

        if graph_format == GraphFormat.BITMASK:
            bitmask = np.zeros((edge_colors - 1, order), dtype=np.int)
        elif graph_format == GraphFormat.ADJACENCY_MATRIX:
            adjacency_matrix = np.zeros((order, order), dtype=int)
        elif graph_format == GraphFormat.FLATTENED_COLUMN_FIRST:
            flattened_column_first = np.zeros((order * (order - 1) // 2,), dtype=int)
        else:
            flattened_row_first = np.zeros((order * (order - 1) // 2,), dtype=int)

        super().__init__(
            graph_format=graph_format,
            edge_colors=edge_colors,
            bitmask=bitmask,
            adjacency_matrix=adjacency_matrix,
            flattened_column_first=flattened_column_first,
            flattened_row_first=flattened_row_first,
        )


class CompleteGraph(Graph):
    """
    This class inherits the `Graph` class and it is used to instantiate complete graphs. In other
    words, it produces a graph of a given order and with two available edge colors such that all of
    its edges are colored in the color 1. It is also possible to configure the starting graph
    format that the instance should be initialized in.
    """

    def __init__(self, graph_format: GraphFormat, order: int):
        """
        This constructor initializes the desired complete graph in the selected graph format.

        :param graph_format: The starting graph format the graph should be initialized in, given as
            an item of the `GraphFormat` enumeration.
        :param order: The graph order, given as a positive integer.
        """

        bitmask = None
        adjacency_matrix = None
        flattened_column_first = None
        flattened_row_first = None

        if graph_format == GraphFormat.BITMASK:
            bitmask = np.zeros((1, order), dtype=np.int)
            bitmask[0, :] = (1 << order) - 1
            bitmask[0, :] -= 1 << np.arange(order, dtype=np.int)

        elif graph_format == GraphFormat.ADJACENCY_MATRIX:
            adjacency_matrix = np.ones((order, order), dtype=int)
            np.fill_diagonal(adjacency_matrix, 0)

        elif graph_format == GraphFormat.FLATTENED_COLUMN_FIRST:
            flattened_column_first = np.ones((order * (order - 1) // 2,), dtype=int)

        else:
            flattened_row_first = np.ones((order * (order - 1) // 2,), dtype=int)

        super().__init__(
            graph_format=graph_format,
            edge_colors=2,
            bitmask=bitmask,
            adjacency_matrix=adjacency_matrix,
            flattened_column_first=flattened_column_first,
            flattened_row_first=flattened_row_first,
        )


class AlmostCompleteGraph(Graph):
    """
    This class inherits the `Graph` class and it is used to instantiate almost-complete graphs. In
    other words, it produces a graph of a given order and with two available edge colors such that
    all of its edges are colored in the color 1, except for the edge between the last two vertices,
    which is colored in the color 0. It is also possible to configure the starting graph format
    that the instance should be initialized in.
    """

    def __init__(self, graph_format: GraphFormat, order: int):
        """
        This constructor initializes the desired almost-complete graph in the selected graph
        format.

        :param graph_format: The starting graph format the graph should be initialized in, given as
            an item of the `GraphFormat` enumeration.
        :param order: The graph order, given as a positive integer (not below two).
        """

        bitmask = None
        adjacency_matrix = None
        flattened_column_first = None
        flattened_row_first = None

        if graph_format == GraphFormat.BITMASK:
            bitmask = np.zeros((1, order), dtype=np.int)
            bitmask[0, :] = (1 << order) - 1
            bitmask[0, :] -= 1 << np.arange(order, dtype=np.int)
            bitmask[0, -1] -= 1 << (order - 2)
            bitmask[0, -2] -= 1 << (order - 1)

        elif graph_format == GraphFormat.ADJACENCY_MATRIX:
            adjacency_matrix = np.ones((order, order), dtype=int)
            np.fill_diagonal(adjacency_matrix, 0)
            adjacency_matrix[-1, -2] = 0
            adjacency_matrix[-2, -1] = 0

        elif graph_format == GraphFormat.FLATTENED_COLUMN_FIRST:
            flattened_column_first = np.ones((order * (order - 1) // 2,), dtype=int)
            flattened_column_first[-1] = 0

        else:
            flattened_row_first = np.ones((order * (order - 1) // 2,), dtype=int)
            flattened_row_first[-1] = 0

        super().__init__(
            graph_format=graph_format,
            edge_colors=2,
            bitmask=bitmask,
            adjacency_matrix=adjacency_matrix,
            flattened_column_first=flattened_column_first,
            flattened_row_first=flattened_row_first,
        )


class CompleteBipartiteGraph(Graph):
    """
    #TODO
    """

    def __init__(self, graph_format: GraphFormat, partition_size_1: int, partition_size_2: int):
        """
        #TODO
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
    """
    #TODO
    """

    def __init__(self, graph_format: GraphFormat, partition_sizes: List[int]):
        """
        #TODO
        """

        order = sum(partition_sizes)
        bitmask = None
        adjacency_matrix = None

        if graph_format == GraphFormat.BITMASK:
            bitmask = np.full((1, order), (1 << order) - 1, dtype=int)

            start = 0
            for item in partition_sizes:
                bitmask[0, start : start + item] -= (1 << (start + item)) - (1 << start)
                start += item
            
        elif graph_format == GraphFormat.ADJACENCY_MATRIX:
            adjacency_matrix = np.ones((order, order), dtype=int)

            start = 0
            for item in partition_sizes:
                adjacency_matrix[start:start + item, start:start + item] = 0
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
    #TODO
    """

    def __init__(self, graph_format: GraphFormat, order: int, central_vertex: int = 0):
        """
        #TODO
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
        
        elif graph_format == GraphFormat.FLATTENED_COLUMN_FIRST:
            flattened_column_first = np.zeros((order * (order - 1) // 2,), dtype=int)
            start = central_vertex * (central_vertex - 1) // 2
            flattened_column_first[start : start + central_vertex] = 1

            indices = np.arange(central_vertex + 1, order, dtype=int)
            indices = indices * (indices - 1) // 2 + central_vertex
            flattened_column_first[indices] = 1
        
        else:
            flattened_row_first = np.zeros((order * (order - 1) // 2,), dtype=int)
            start = central_vertex * (2 * order - 1 - central_vertex) // 2
            flattened_row_first[start : start + order - central_vertex - 1] = 1

            indices = np.arange(1, central_vertex + 1, dtype=int)
            indices = indices * (2 * order - 3 - indices) // 2 + central_vertex
            flattened_row_first[indices] = 1

        super().__init__(
            graph_format=graph_format,
            edge_colors=2,
            bitmask=bitmask,
            adjacency_matrix=adjacency_matrix,
            flattened_column_first=flattened_column_first,
            flattened_row_first=flattened_row_first,
        )


class PathGraph(Graph):
    """
    #TODO
    """

    def __init__(self, graph_format: GraphFormat, order: int):
        """
        #TODO
        """

        bitmask = None
        adjacency_matrix = None
        flattened_column_first = None
        flattened_row_first = None

        if graph_format == GraphFormat.BITMASK:
            bitmask = ((1 << np.arange(order, dtype=np.int)) * 5 // 2).reshape(1, -1)
            bitmask[0, 0] = 2
            bitmask[0, -1] = 1 << (order - 2)
        elif graph_format == GraphFormat.ADJACENCY_MATRIX:
            adjacency_matrix = np.zeros((order, order), dtype=int)
            rows = np.arange(order - 1, dtype = int)
            adjacency_matrix[rows, rows + 1] = 1
            adjacency_matrix[rows + 1, rows] = 1
        elif graph_format == GraphFormat.FLATTENED_COLUMN_FIRST:
            pass


def path_graph_adjacencies(order: int) -> np.ndarray:
    graph_adjacencies = (1 << np.arange(order, dtype=np.uint64)) * 5 // 2
    graph_adjacencies[0] = 2
    graph_adjacencies[-1] = 1 << (order - 2)

    return graph_adjacencies


def cycle_graph_adjacencies(order: int) -> np.ndarray:
    graph_adjacencies = (1 << np.arange(order, dtype=np.uint64)) * 5 // 2
    graph_adjacencies[0] = (1 << (order - 1)) + 2
    graph_adjacencies[-1] = (1 << (order - 2)) + 1

    return graph_adjacencies


def wheel_graph_adjacencies(order: int) -> np.ndarray:
    graph_adjacencies = (1 << np.arange(order, dtype=np.uint64)) * 5 // 2 + 1
    graph_adjacencies[0] = (1 << order) - 2
    graph_adjacencies[1] = (1 << (order - 1)) + 5
    graph_adjacencies[-1] = (1 << (order - 2)) + 3

    return graph_adjacencies


def book_graph_adjacencies(index: int) -> np.ndarray:
    graph_adjacencies = np.full((index + 2,), 3, dtype=np.uint64)
    graph_adjacencies[0] = (1 << (index + 2)) - 2
    graph_adjacencies[1] = (1 << (index + 2)) - 3

    return graph_adjacencies


def friendship_graph_adjacencies(index: int) -> np.ndarray:
    graph_adjacencies = np.full((2 * index + 1,), 1, dtype=np.uint64)
    graph_adjacencies[0] = (1 << (2 * index + 1)) - 2
    graph_adjacencies[1::2] += 1 << np.arange(2, 2 * index + 2, 2, dtype=np.uint64)
    graph_adjacencies[2::2] += 1 << np.arange(1, 2 * index + 1, 2, dtype=np.uint64)

    return graph_adjacencies
