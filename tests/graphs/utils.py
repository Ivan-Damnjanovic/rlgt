"""
This testing module contains two utility functions to be used by the functions that directly test
the functionalities from the `rl_graph_theory.graphs` package.
"""

from typing import Callable

import numpy as np


def verify_instantiated_graph(
    constructor: Callable,
    edge_colors: int,
    order: int,
    bitmask_in: np.ndarray,
    bitmask_out: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_clockwise: np.ndarray,
    flattened_row_major: np.ndarray,
    is_directed: bool = False,
    allow_loops: bool = False,
):
    """
    This function verifies whether a given constructor correctly instantiates a `Graph` object. It
    accepts a function without arguments that instantiates a `Graph` object, and checks whether
    this object correctly behaves and represents the desired graph in all the graph formats. An
    `AssertionError` is raised if the function detects an error in the constructed object.

    :param constructor: The function without arguments that instantiates a `Graph` object, which
        should be tested.
    :param edge_colors: The correct number of available edge colors of the desired graph, given as
        an `int`.
    :param order: The correct desired graph order, given as an `int`.
    :param bitmask_in: The correct `np.ndarray` that represents the desired graph in the bitmask
        in-neighborhoods format.
    :param bitmask_out: The correct `np.ndarray` that represents the desired graph in the bitmask
        out-neighborhoods format.
    :param adjacency_matrix: The correct `np.ndarray` that represents the desired graph in the
        adjacency matrix format.
    :param flattened_column_first: The correct `np.ndarray` that represents the desired graph in
        the flattened column-first format.
    :param flattened_row_first: The correct `np.ndarray` that represents the desired graph in the
        flattened row-first format.
    """

    instance = constructor()
    assert instance.edge_colors == edge_colors, f"{instance.edge_colors}, {edge_colors}"
    assert instance.order == order, f"{instance.order}, {order}"
    assert instance.is_directed == is_directed, f"{instance.is_directed}, {is_directed}"
    assert instance.allow_loops == allow_loops, f"{instance.allow_loops}, {allow_loops}"

    instance = constructor()
    np.testing.assert_array_equal(instance.bitmask_in, bitmask_in)
    np.testing.assert_array_equal(instance._Graph__bitmask_in, bitmask_in)

    instance = constructor()
    np.testing.assert_array_equal(instance.bitmask_out, bitmask_out)
    np.testing.assert_array_equal(instance._Graph__bitmask_out, bitmask_out)

    instance = constructor()
    np.testing.assert_array_equal(instance.adjacency_matrix, adjacency_matrix)
    np.testing.assert_array_equal(instance._Graph__adjacency_matrix, adjacency_matrix)

    instance = constructor()
    np.testing.assert_array_equal(instance.flattened_clockwise_colors, flattened_clockwise)
    np.testing.assert_array_equal(instance._Graph__flattened_clockwise, flattened_clockwise)

    instance = constructor()
    np.testing.assert_array_equal(instance.flattened_row_major_colors, flattened_row_major)
    np.testing.assert_array_equal(instance._Graph__flattened_row_major, flattened_row_major)


def verify_instantiated_graph_batch(
    constructor: Callable,
    batch_size: int,
    edge_colors: int,
    order: int,
    bitmask_in: np.ndarray,
    bitmask_out: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_clockwise: np.ndarray,
    flattened_row_major: np.ndarray,
    is_directed: bool = False,
    allow_loops: bool = False,
):
    """
    This function verifies whether a given constructor correctly instantiates a `GraphBatch`
    object. It accepts a function without arguments that instantiates a `GraphBatch` object, and
    checks whether this object correctly behaves and represents the desired batch of graphs in all
    the graph formats. An `AssertionError` is raised if the function detects an error in the
    constructed object.

    :param constructor: The function without arguments that instantiates a `GraphBatch` object,
        which should be tested.
    :param batch_size: The correct batch size of the desired batch of graphs, given as an `int`.
    :param edge_colors: The correct number of available edge colors of the desired batch of graphs,
        given as an `int`.
    :param order: The correct order of all the graphs from the desired batch of graphs, given as an
        `int`.
    :param bitmask_batch: The correct `np.ndarray` that represents the desired batch of graphs in
        the bitmask format.
    :param adjacency_matrix_batch: The correct `np.ndarray` that represents the desired batch of
        graphs in the adjacency matrix format.
    :param flattened_column_first_batch: The correct `np.ndarray` that represents the desired batch
        of graphs in the flattened column-first format.
    :param flattened_row_first_batch: The correct `np.ndarray` that represents the desired batch of
        graphs in the flattened row-first format.
    """

    instance = constructor()
    assert instance.batch_size == batch_size, f"{instance.batch_size}, {batch_size}"
    assert instance.edge_colors == edge_colors, f"{instance.edge_colors}, {edge_colors}"
    assert instance.order == order, f"{instance.order}, {order}"
    assert instance.is_directed == is_directed, f"{instance.is_directed}, {is_directed}"
    assert instance.allow_loops == allow_loops, f"{instance.allow_loops}, {allow_loops}"

    instance = constructor()
    np.testing.assert_array_equal(instance.bitmask_in, bitmask_in)
    np.testing.assert_array_equal(instance._GraphBatch__bitmask_in, bitmask_in)

    instance = constructor()
    np.testing.assert_array_equal(instance.bitmask_out, bitmask_out)
    np.testing.assert_array_equal(instance._GraphBatch__bitmask_out, bitmask_out)

    instance = constructor()
    np.testing.assert_array_equal(instance.adjacency_matrix, adjacency_matrix)
    np.testing.assert_array_equal(instance._GraphBatch__adjacency_matrix, adjacency_matrix)

    instance = constructor()
    np.testing.assert_array_equal(instance.flattened_clockwise_colors, flattened_clockwise)
    np.testing.assert_array_equal(instance._GraphBatch__flattened_clockwise, flattened_clockwise)

    instance = constructor()
    np.testing.assert_array_equal(instance.flattened_row_major_colors, flattened_row_major)
    np.testing.assert_array_equal(instance._GraphBatch__flattened_row_major, flattened_row_major)
