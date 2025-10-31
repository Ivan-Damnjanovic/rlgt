import numpy as np
import pytest

from rl_graph_theory.graphs.graph import EdgeOrdering, Graph


EXAMPLE_GRAPHS = [
    (
        4,
        7,
        np.array(
            [
                [6, 21, 35, 0, 2, 4, 0],
                [32, 32, 0, 0, 0, 3, 0],
                [16, 8, 16, 18, 109, 16, 16],
                [72, 64, 0, 1, 0, 0, 3],
            ],
            dtype=int,
        ),
        np.array(
            [
                [0, 0, 0, 3, 2, 1, 3],
                [0, 0, 0, 2, 0, 1, 3],
                [0, 0, 0, 4, 2, 0, 4],
                [3, 2, 4, 0, 2, 4, 4],
                [2, 0, 2, 2, 0, 2, 2],
                [1, 1, 0, 4, 2, 0, 4],
                [3, 3, 4, 4, 2, 4, 0],
            ],
            dtype=int,
        ),
        np.array([0, 0, 0, 3, 2, 4, 2, 0, 2, 2, 1, 1, 0, 4, 2, 3, 3, 4, 4, 2, 4], dtype=int),
        np.array([0, 0, 3, 2, 1, 3, 0, 2, 0, 1, 3, 4, 2, 0, 4, 2, 4, 4, 2, 2, 4], dtype=int),
    ),
    (
        3,
        8,
        np.array(
            [
                [32, 0, 16, 32, 4, 9, 0, 0],
                [80, 128, 40, 68, 65, 68, 57, 2],
                [0, 16, 192, 144, 170, 16, 132, 92],
            ],
            dtype=int,
        ),
        np.array(
            [
                [0, 3, 3, 3, 1, 0, 1, 3],
                [3, 0, 3, 3, 2, 3, 3, 1],
                [3, 3, 0, 1, 0, 1, 2, 2],
                [3, 3, 1, 0, 2, 0, 1, 2],
                [1, 2, 0, 2, 0, 2, 1, 2],
                [0, 3, 1, 0, 2, 0, 1, 3],
                [1, 3, 2, 1, 1, 1, 0, 2],
                [3, 1, 2, 2, 2, 3, 2, 0],
            ],
            dtype=int,
        ),
        np.array(
            [3, 3, 3, 3, 3, 1, 1, 2, 0, 2, 0, 3, 1, 0, 2, 1, 3, 2, 1, 1, 1, 3, 1, 2, 2, 2, 3, 2],
            dtype=int,
        ),
        np.array(
            [3, 3, 3, 1, 0, 1, 3, 3, 3, 2, 3, 3, 1, 1, 0, 1, 2, 2, 2, 0, 1, 2, 2, 1, 2, 1, 3, 2],
            dtype=int,
        ),
    ),
]


@pytest.mark.parametrize(
    "edge_colors, order, bitmask_format, adjacency_matrix, flattened_column_first, "
    "flattened_row_first",
    EXAMPLE_GRAPHS,
)
def test_graph_bitmask_format(
    edge_colors: int,
    order: int,
    bitmask_format: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_column_first: np.ndarray,
    flattened_row_first: np.ndarray,
):
    """
    #TODO
    """

    example = Graph.from_bitmask_format(bitmask_format)
    assert example.edge_colors == edge_colors
    assert example.order == order
    assert np.all(example.bitmask_format == bitmask_format)

    example = Graph.from_bitmask_format(bitmask_format)
    assert np.all(example.adjacency_matrix == adjacency_matrix)
    example = Graph.from_bitmask_format(bitmask_format)
    assert np.all(example.flattened_column_first == flattened_column_first)
    example = Graph.from_bitmask_format(bitmask_format)
    assert np.all(example.flattened_row_first == flattened_row_first)


@pytest.mark.parametrize(
    "edge_colors, order, bitmask_format, adjacency_matrix, flattened_column_first, "
    "flattened_row_first",
    EXAMPLE_GRAPHS,
)
def test_graph_adjacency_matrix(
    edge_colors: int,
    order: int,
    bitmask_format: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_column_first: np.ndarray,
    flattened_row_first: np.ndarray,
):
    """
    #TODO
    """

    example = Graph.from_adjacency_matrix(adjacency_matrix, edge_colors)
    assert example.edge_colors == edge_colors
    assert example.order == order
    assert np.all(example.adjacency_matrix == adjacency_matrix)

    example = Graph.from_adjacency_matrix(adjacency_matrix, edge_colors)
    assert np.all(example.bitmask_format == bitmask_format)
    example = Graph.from_adjacency_matrix(adjacency_matrix, edge_colors)
    assert np.all(example.flattened_column_first == flattened_column_first)
    example = Graph.from_adjacency_matrix(adjacency_matrix, edge_colors)
    assert np.all(example.flattened_row_first == flattened_row_first)


@pytest.mark.parametrize(
    "edge_colors, order, bitmask_format, adjacency_matrix, flattened_column_first, "
    "flattened_row_first",
    EXAMPLE_GRAPHS,
)
def test_graph_flattened_column_first(
    edge_colors: int,
    order: int,
    bitmask_format: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_column_first: np.ndarray,
    flattened_row_first: np.ndarray,
):
    """
    #TODO
    """

    example = Graph.from_flattened_format(
        flattened_column_first, EdgeOrdering.COLUMN_FIRST, edge_colors
    )
    assert example.edge_colors == edge_colors
    assert example.order == order
    assert np.all(example.flattened_column_first == flattened_column_first)

    example = Graph.from_flattened_format(
        flattened_column_first, EdgeOrdering.COLUMN_FIRST, edge_colors
    )
    assert np.all(example.bitmask_format == bitmask_format)
    example = Graph.from_flattened_format(
        flattened_column_first, EdgeOrdering.COLUMN_FIRST, edge_colors
    )
    assert np.all(example.adjacency_matrix == adjacency_matrix)
    example = Graph.from_flattened_format(
        flattened_column_first, EdgeOrdering.COLUMN_FIRST, edge_colors
    )
    assert np.all(example.flattened_row_first == flattened_row_first)


@pytest.mark.parametrize(
    "edge_colors, order, bitmask_format, adjacency_matrix, flattened_column_first, "
    "flattened_row_first",
    EXAMPLE_GRAPHS,
)
def test_graph_flattened_row_first(
    edge_colors: int,
    order: int,
    bitmask_format: np.ndarray,
    adjacency_matrix: np.ndarray,
    flattened_column_first: np.ndarray,
    flattened_row_first: np.ndarray,
):
    """
    #TODO
    """

    example = Graph.from_flattened_format(flattened_row_first, EdgeOrdering.ROW_FIRST, edge_colors)
    assert example.edge_colors == edge_colors
    assert example.order == order
    assert np.all(example.flattened_row_first == flattened_row_first)

    example = Graph.from_flattened_format(flattened_row_first, EdgeOrdering.ROW_FIRST, edge_colors)
    assert np.all(example.bitmask_format == bitmask_format)
    example = Graph.from_flattened_format(flattened_row_first, EdgeOrdering.ROW_FIRST, edge_colors)
    assert np.all(example.adjacency_matrix == adjacency_matrix)
    example = Graph.from_flattened_format(flattened_row_first, EdgeOrdering.ROW_FIRST, edge_colors)
    assert np.all(example.flattened_column_first == flattened_column_first)
