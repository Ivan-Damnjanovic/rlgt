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
    bitmask: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_column_first: np.ndarray,
    flattened_row_first: np.ndarray,
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
    :param bitmask: The correct `np.ndarray` that represents the desired graph in the bitmask
        format.
    :param adjacency_matrix: The correct `np.ndarray` that represents the desired graph in the
        adjacency matrix format.
    :param flattened_column_first: The correct `np.ndarray` that represents the desired graph in
        the flattened column-first format.
    :param flattened_row_first: The correct `np.ndarray` that represents the desired graph in the
        flattened row-first format.
    """

    instance = constructor()
    assert instance.edge_colors == edge_colors
    assert instance.order == order
    assert instance.bitmask.shape == bitmask.shape
    assert np.all(instance.bitmask == bitmask)
    assert instance._Graph__bitmask.shape == bitmask.shape
    assert np.all(instance._Graph__bitmask == bitmask)

    instance = constructor()
    assert instance.adjacency_matrix.shape == adjacency_matrix.shape
    assert np.all(instance.adjacency_matrix == adjacency_matrix)
    assert instance._Graph__adjacency_matrix.shape == adjacency_matrix.shape
    assert np.all(instance._Graph__adjacency_matrix == adjacency_matrix)

    instance = constructor()
    assert instance.flattened_column_first.shape == flattened_column_first.shape
    assert np.all(instance.flattened_column_first == flattened_column_first)
    assert instance._Graph__flattened_column_first.shape == flattened_column_first.shape
    assert np.all(instance._Graph__flattened_column_first == flattened_column_first)

    instance = constructor()
    assert instance.flattened_row_first.shape == flattened_row_first.shape
    assert np.all(instance.flattened_row_first == flattened_row_first)
    assert instance._Graph__flattened_row_first.shape == flattened_row_first.shape
    assert np.all(instance._Graph__flattened_row_first == flattened_row_first)


def verify_instantiated_graph_batch(
    constructor: Callable,
    batch_size: int,
    edge_colors: int,
    order: int,
    bitmask_batch: np.ndarray,
    adjacency_matrix_batch: np.ndarray,
    flattened_column_first_batch: np.ndarray,
    flattened_row_first_batch: np.ndarray,
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
    assert instance.batch_size == batch_size
    assert instance.edge_colors == edge_colors
    assert instance.order == order
    assert instance.bitmask_batch.shape == bitmask_batch.shape
    assert np.all(instance.bitmask_batch == bitmask_batch)
    assert instance._GraphBatch__bitmask_batch.shape == bitmask_batch.shape
    assert np.all(instance._GraphBatch__bitmask_batch == bitmask_batch)

    instance = constructor()
    assert instance.adjacency_matrix_batch.shape == adjacency_matrix_batch.shape
    assert np.all(instance.adjacency_matrix_batch == adjacency_matrix_batch)
    assert instance._GraphBatch__adjacency_matrix_batch.shape == adjacency_matrix_batch.shape
    assert np.all(instance._GraphBatch__adjacency_matrix_batch == adjacency_matrix_batch)

    instance = constructor()
    assert instance.flattened_column_first_batch.shape == flattened_column_first_batch.shape
    assert np.all(instance.flattened_column_first_batch == flattened_column_first_batch)
    assert (
        instance._GraphBatch__flattened_column_first_batch.shape
        == flattened_column_first_batch.shape
    )
    assert np.all(
        instance._GraphBatch__flattened_column_first_batch == flattened_column_first_batch
    )

    instance = constructor()
    assert instance.flattened_row_first_batch.shape == flattened_row_first_batch.shape
    assert np.all(instance.flattened_row_first_batch == flattened_row_first_batch)
    assert instance._GraphBatch__flattened_row_first_batch.shape == flattened_row_first_batch.shape
    assert np.all(instance._GraphBatch__flattened_row_first_batch == flattened_row_first_batch)
