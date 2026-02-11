import pytest
from typing import Any
import numpy as np

from rlgt.graphs.graph import Graph

from .graph_test_cases import TEST_CASES_CONSTRUCTOR


@pytest.mark.parametrize(
    "graph_order, edge_colors, is_directed, allow_loops, kwarg",
    TEST_CASES_CONSTRUCTOR,
)
def test_constructor(
    graph_order: int,
    edge_colors: int,
    is_directed: bool,
    allow_loops: bool,
    kwarg: dict[str, np.ndarray],
):
    g = Graph(edge_colors=edge_colors, is_directed=is_directed, allow_loops=allow_loops, **kwarg)
    assert g.graph_order == graph_order


@pytest.mark.parametrize(
    "kwargs, result",
    [
        (
            {"bitmask_out": np.asarray([[2, 1]], np.uint8)},
            np.asarray([[2, 1]], np.uint8),
        ),
        (
            {"adjacency_matrix_colors": np.asarray([[0, 1], [1, 0]], np.uint8)},
            np.asarray([[2, 1]], np.uint8),
        ),
    ],
)
def test_bitmask_out(kwargs: dict[str, Any], result: np.ndarray):
    graph = Graph(**kwargs)
    np.testing.assert_array_equal(graph.bitmask_out, result)


@pytest.mark.parametrize(
    "kwargs, result",
    [
        (
            {"bitmask_in": np.asarray([[2, 1]], np.uint8)},
            np.asarray([[2, 1]], np.uint8),
        ),
        (
            {"adjacency_matrix_colors": np.asarray([[0, 1], [1, 0]], np.uint8)},
            np.asarray([[2, 1]], np.uint8),
        ),
    ],
)
def test_bitmask_in(kwargs: dict[str, Any], result: np.ndarray):
    graph = Graph(**kwargs)
    np.testing.assert_array_equal(graph.bitmask_in, result)


@pytest.mark.parametrize(
    "kwargs, result",
    [
        (
            {"adjacency_matrix_colors": np.asarray([[0, 1], [1, 0]], np.uint8)},
            np.asarray([[0, 1], [1, 0]], np.uint8),
        ),
        (
            {"adjacency_matrix_binary": np.asarray([[[0, 1], [1, 0]]], np.uint8)},
            np.asarray([[0, 1], [1, 0]], np.uint8),
        ),
        (
            {"flattened_row_major_colors": np.asarray([1], np.uint8)},
            np.asarray([[0, 1], [1, 0]], np.uint8),
        ),
        (
            {"flattened_clockwise_colors": np.asarray([1], np.uint8)},
            np.asarray([[0, 1], [1, 0]], np.uint8),
        ),
        (
            {"flattened_row_major_binary": np.asarray([[1]], np.uint8)},
            np.asarray([[0, 1], [1, 0]], np.uint8),
        ),
        (
            {"flattened_clockwise_binary": np.asarray([[1]], np.uint8)},
            np.asarray([[0, 1], [1, 0]], np.uint8),
        ),
    ],
)
def test_adjacency_matrix_colors(kwargs: dict[str, Any], result: np.ndarray):
    graph = Graph(**kwargs)
    np.testing.assert_array_equal(graph.adjacency_matrix_colors, result)


@pytest.mark.parametrize(
    "kwargs, result",
    [
        (
            {"adjacency_matrix_binary": np.asarray([[[0, 1], [1, 0]]], np.uint8)},
            np.asarray([[[0, 1], [1, 0]]], np.uint8),
        ),
        (
            {"bitmask_out": np.asarray([[2, 1]], np.uint8)},
            np.asarray([[[0, 1], [1, 0]]], np.uint8),
        ),
        (
            {"bitmask_in": np.asarray([[2, 1]], np.uint8), "is_directed": True},
            np.asarray([[[0, 1], [1, 0]]], np.uint8),
        ),
        (
            {"flattened_row_major_colors": np.asarray([1], np.uint8)},
            np.asarray([[[0, 1], [1, 0]]], np.uint8),
        ),
        (
            {"flattened_clockwise_colors": np.asarray([1], np.uint8)},
            np.asarray([[[0, 1], [1, 0]]], np.uint8),
        ),
        (
            {"flattened_row_major_binary": np.asarray([[1]], np.uint8)},
            np.asarray([[[0, 1], [1, 0]]], np.uint8),
        ),
        (
            {"flattened_clockwise_binary": np.asarray([[1]], np.uint8)},
            np.asarray([[[0, 1], [1, 0]]], np.uint8),
        ),
    ],
)
def test_adjacency_matrix_binary(kwargs: dict[str, Any], result: np.ndarray):
    graph = Graph(**kwargs)
    np.testing.assert_array_equal(graph.adjacency_matrix_binary, result)


@pytest.mark.parametrize(
    "kwargs, result",
    [
        (
            {"flattened_row_major_colors": np.asarray([1], np.uint8)},
            np.asarray([1], np.uint8),
        ),
        (
            {"flattened_row_major_binary": np.asarray([[1]], np.uint8)},
            np.asarray([1], np.uint8),
        ),
        (
            {"adjacency_matrix_binary": np.asarray([[[0, 1], [1, 0]]], np.uint8)},
            np.asarray([1], np.uint8),
        ),
    ],
)
def test_flattened_row_major_colors(kwargs: dict[str, Any], result: np.ndarray):
    graph = Graph(**kwargs)
    np.testing.assert_array_equal(graph.flattened_row_major_colors, result)


@pytest.mark.parametrize(
    "kwargs, result",
    [
        (
            {"flattened_clockwise_colors": np.asarray([1], np.uint8)},
            np.asarray([1], np.uint8),
        ),
        (
            {"flattened_clockwise_binary": np.asarray([[1]], np.uint8)},
            np.asarray([1], np.uint8),
        ),
        (
            {"adjacency_matrix_binary": np.asarray([[[0, 1], [1, 0]]], np.uint8)},
            np.asarray([1], np.uint8),
        ),
    ],
)
def test_flattened_clockwise_colors(kwargs: dict[str, Any], result: np.ndarray):
    graph = Graph(**kwargs)
    np.testing.assert_array_equal(graph.flattened_clockwise_colors, result)


@pytest.mark.parametrize(
    "kwargs, result",
    [
        (
            {"flattened_row_major_binary": np.asarray([[1]], np.uint8)},
            np.asarray([[1]], np.uint8),
        ),
        (
            {"flattened_row_major_colors": np.asarray([1], np.uint8)},
            np.asarray([[1]], np.uint8),
        ),
        (
            {"flattened_clockwise_colors": np.asarray([1], np.uint8)},
            np.asarray([[1]], np.uint8),
        ),
        (
            {"adjacency_matrix_colors": np.asarray([[0, 1], [1, 0]], np.uint8)},
            np.asarray([[1]], np.uint8),
        ),
        (
            {"adjacency_matrix_binary": np.asarray([[[0, 1], [1, 0]]], np.uint8)},
            np.asarray([[1]], np.uint8),
        ),
    ],
)
def test_flattened_row_major_binary(kwargs: dict[str, Any], result: np.ndarray):
    graph = Graph(**kwargs)
    np.testing.assert_array_equal(graph.flattened_row_major_binary, result)


@pytest.mark.parametrize(
    "kwargs, result",
    [
        (
            {"flattened_clockwise_binary": np.asarray([[1]], np.uint8)},
            np.asarray([[1]], np.uint8),
        ),
        (
            {"flattened_row_major_colors": np.asarray([1], np.uint8)},
            np.asarray([[1]], np.uint8),
        ),
        (
            {"flattened_clockwise_colors": np.asarray([1], np.uint8)},
            np.asarray([[1]], np.uint8),
        ),
        (
            {"adjacency_matrix_colors": np.asarray([[0, 1], [1, 0]], np.uint8)},
            np.asarray([[1]], np.uint8),
        ),
        (
            {"adjacency_matrix_binary": np.asarray([[[0, 1], [1, 0]]], np.uint8)},
            np.asarray([[1]], np.uint8),
        ),
    ],
)
def test_flattened_clockwise_binary(kwargs: dict[str, Any], result: np.ndarray):
    graph = Graph(**kwargs)
    np.testing.assert_array_equal(graph.flattened_clockwise_binary, result)
