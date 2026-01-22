"""
This testing module contains two utility functions to be used by the functions that directly test
the functionalities from the `rl_graph_theory.graphs` package.
"""

from typing import Callable

import numpy as np

from rl_graph_theory.graphs.graph import Graph


def verify_instantiated_graph(
    constructor: Callable[[], Graph],
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
    # assert instance.edge_colors == edge_colors, f"{instance.edge_colors}, {edge_colors}"
    assert instance.edge_colors == edge_colors, f"{instance.edge_colors}, {edge_colors}"
    assert instance.graph_order == order, f"{instance.order}, {order}"
    assert instance.is_directed == is_directed, f"{instance.is_directed}, {is_directed}"
    assert instance.allow_loops == allow_loops, f"{instance.allow_loops}, {allow_loops}"

    instance = constructor()
    assert_bitmask_equal(instance.bitmask_in, bitmask_in)
    assert_bitmask_equal(instance._Graph__bitmask_in, bitmask_in)

    instance = constructor()
    assert_bitmask_equal(instance.bitmask_out, bitmask_out)
    assert_bitmask_equal(instance._Graph__bitmask_out, bitmask_out)

    instance = constructor()
    np.testing.assert_array_equal(instance.adjacency_matrix_colors, adjacency_matrix)
    np.testing.assert_array_equal(instance._Graph__adjacency_matrix_colors, adjacency_matrix)

    instance = constructor()
    np.testing.assert_array_equal(instance.flattened_clockwise_colors, flattened_clockwise)
    np.testing.assert_array_equal(instance._Graph__flattened_clockwise_colors, flattened_clockwise)

    instance = constructor()
    np.testing.assert_array_equal(instance.flattened_row_major_colors, flattened_row_major)
    np.testing.assert_array_equal(instance._Graph__flattened_row_major_colors, flattened_row_major)


def assert_bitmask_equal(actual, desired):
    if actual.shape[0] == desired.shape[0] + 1:
        pad_shape = list(desired.shape)
        pad_shape[0] = 1
        desired = np.concat([desired, np.zeros(pad_shape, np.uint8)], axis=0)

    np.testing.assert_array_equal(actual, desired)
