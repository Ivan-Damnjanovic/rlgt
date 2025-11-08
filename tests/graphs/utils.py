"""
#TODO
"""

import numpy as np
from typing import Callable


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
    #TODO
    """

    instance = constructor()
    assert instance.edge_colors == edge_colors
    assert instance.order == order
    assert np.all(instance.bitmask == bitmask)
    assert np.all(instance._Graph__bitmask == bitmask)

    instance = constructor()
    assert np.all(instance.adjacency_matrix == adjacency_matrix)
    assert np.all(instance._Graph__adjacency_matrix == adjacency_matrix)

    instance = constructor()
    assert np.all(instance.flattened_column_first == flattened_column_first)
    assert np.all(instance._Graph__flattened_column_first == flattened_column_first)

    instance = constructor()
    assert np.all(instance.flattened_row_first == flattened_row_first)
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
    #TODO
    """

    instance = constructor()
    assert instance.batch_size == batch_size
    assert instance.edge_colors == edge_colors
    assert instance.order == order
    assert np.all(instance.bitmask_batch == bitmask_batch)
    assert np.all(instance._GraphBatch__bitmask_batch == bitmask_batch)

    instance = constructor()
    assert np.all(instance.adjacency_matrix_batch == adjacency_matrix_batch)
    assert np.all(instance._GraphBatch__adjacency_matrix_batch == adjacency_matrix_batch)

    instance = constructor()
    assert np.all(instance.flattened_column_first_batch == flattened_column_first_batch)
    assert np.all(instance._GraphBatch__flattened_column_first_batch == flattened_column_first_batch)

    instance = constructor()
    assert np.all(instance.flattened_row_first_batch == flattened_row_first_batch)
    assert np.all(instance._GraphBatch__flattened_row_first_batch == flattened_row_first_batch)