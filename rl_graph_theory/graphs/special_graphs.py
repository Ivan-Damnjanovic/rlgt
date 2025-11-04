"""
#TODO
"""

import numpy as np
from typing import List

from graph import Graph, GraphFormat


class CompleteGraph(Graph):
    """
    #TODO
    """

    def __init__(self, graph_format: GraphFormat, order: int):
        """
        #TODO
        """

        



def complete_graph_adjacencies(order: int) -> np.ndarray:
    graph_adjacencies = np.full((order,), (1 << order) - 1, dtype=np.uint64)
    graph_adjacencies -= 1 << np.arange(order, dtype=np.uint64)

    return graph_adjacencies


def almost_complete_graph_adjacencies(order: int) -> np.ndarray:
    graph_adjacencies = np.full((order,), (1 << order) - 1, dtype=np.uint64)
    graph_adjacencies -= 1 << np.arange(order, dtype=np.uint64)
    graph_adjacencies[-1] -= 1 << (order - 2)
    graph_adjacencies[-2] -= 1 << (order - 1)

    return graph_adjacencies


def complete_bipartite_graph_adjacencies(
    partition_size_1: int, partition_size_2: int
) -> np.ndarray:
    graph_adjacencies = np.full(
        (partition_size_1 + partition_size_2,), (1 << partition_size_1) - 1, dtype=np.uint64
    )
    graph_adjacencies[:partition_size_1] = (1 << (partition_size_1 + partition_size_2)) - (
        1 << partition_size_1
    )

    return graph_adjacencies


def complete_k_partite_graph_adjacencies(partition_sizes: List[int]) -> np.ndarray:
    order = sum(partition_sizes)
    graph_adjacencies = np.full((order,), (1 << order) - 1, dtype=np.uint64)

    start = 0
    for item in partition_sizes:
        graph_adjacencies[start : start + item] -= (1 << (start + item)) - (1 << start)
        start += item

    return graph_adjacencies


def star_graph_adjacencies(order: int) -> np.ndarray:
    graph_adjacencies = np.full((order,), 1, dtype=np.uint64)
    graph_adjacencies[0] = (1 << order) - 2

    return graph_adjacencies


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