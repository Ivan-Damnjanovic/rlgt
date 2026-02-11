import numpy as np
import pytest

from rlgt.environments.graph_generators import (
    create_choose_two_graph_generator,
    create_edge_perturbation_graph_generator,
    create_fixed_graph_generator,
    create_random_graph_generator,
)
from rlgt.graphs.graph import FlattenedOrdering, Graph, GraphFormat

GRAPHS = [
    Graph.from_flattened(
        np.asarray([1], dtype=np.uint8),
        FlattenedOrdering.ROW_MAJOR,
        edge_colors=2,
        is_directed=False,
        allow_loops=False,
    ),
    Graph.from_flattened(
        np.asarray([1, 0, 1], dtype=np.uint8),
        FlattenedOrdering.ROW_MAJOR,
        edge_colors=2,
        is_directed=False,
        allow_loops=False,
    ),
    Graph.from_flattened(
        np.asarray([1, 0, 1, 0, 1, 0], dtype=np.uint8),
        FlattenedOrdering.ROW_MAJOR,
        edge_colors=2,
        is_directed=False,
        allow_loops=False,
    ),
    Graph.from_flattened(
        np.asarray([1, 0, 1, 0, 1, 0], dtype=np.uint8),
        FlattenedOrdering.ROW_MAJOR,
        edge_colors=2,
        is_directed=True,
        allow_loops=False,
    ),
    Graph.from_flattened(
        np.asarray([1, 0, 1, 0, 1, 0], dtype=np.uint8),
        FlattenedOrdering.ROW_MAJOR,
        edge_colors=2,
        is_directed=False,
        allow_loops=True,
    ),
    Graph.from_flattened(
        np.asarray([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8),
        FlattenedOrdering.ROW_MAJOR,
        edge_colors=2,
        is_directed=True,
        allow_loops=True,
    ),
    Graph.from_flattened(
        np.asarray([1, 2, 1], dtype=np.uint8),
        FlattenedOrdering.ROW_MAJOR,
        edge_colors=3,
        is_directed=False,
        allow_loops=False,
    ),
]


def combinations(a, b):
    for _a in a:
        for _b in b:
            yield (_a, _b)


def verify_generator(generator, graphs):
    gb = generator(len(graphs))
    np.testing.assert_array_equal(
        gb.flattened_row_major_colors,
        np.stack([g.flattened_row_major_colors for g in graphs], axis=0),
    )

    assert gb.edge_colors == graphs[0].edge_colors
    assert gb.is_directed == graphs[0].is_directed
    assert gb.allow_loops == graphs[0].allow_loops


@pytest.mark.parametrize(
    "fixed_graph, graph_format",
    list(combinations(GRAPHS, GraphFormat)),
)
def test_fixed(fixed_graph: Graph, graph_format: GraphFormat):
    generator = create_fixed_graph_generator(fixed_graph, graph_format)
    verify_generator(generator, [fixed_graph])
    verify_generator(generator, [fixed_graph] * 3)


@pytest.mark.parametrize(
    "graph, prob",
    zip(
        list(combinations(GRAPHS, GraphFormat)),
        np.arange(0.0, 1.0, 1.0 / (len(GRAPHS) * len(GraphFormat))),
    ),
)
def test_choose_two(graph: tuple[Graph, GraphFormat], prob: float):
    first_graph, graph_format = graph

    second_graph = Graph.from_flattened(
        flattened=first_graph.flattened_row_major_colors,
        edge_colors=first_graph.edge_colors,
        is_directed=first_graph.is_directed,
        allow_loops=first_graph.allow_loops,
    )

    generator = create_choose_two_graph_generator(
        first_graph,
        second_graph,
        prob,
        graph_format,
        np.random.default_rng(42),
    )

    verify_generator(
        generator,
        [
            second_graph if t else first_graph
            for t in np.random.default_rng(42).random(size=(10,)) < prob
        ],
    )


@pytest.mark.parametrize(
    "graph, edge_prob, color_prob, ordering, batch_size, result",
    [
        (
            Graph.from_flattened(
                np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8),
                FlattenedOrdering.ROW_MAJOR,
                edge_colors=2,
                is_directed=True,
                allow_loops=True,
            ),
            0.0,
            None,
            FlattenedOrdering.CLOCKWISE,
            3,
            np.asarray(
                [
                    [0, 0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1, 1],
                ],
                np.uint8,
            ),
        ),
        (
            Graph.from_flattened(
                np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8),
                FlattenedOrdering.ROW_MAJOR,
                edge_colors=2,
                is_directed=True,
                allow_loops=True,
            ),
            0.0,
            None,
            FlattenedOrdering.ROW_MAJOR,
            3,
            np.asarray(
                [
                    [0, 0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1, 1],
                ],
                np.uint8,
            ),
        ),
        (
            Graph.from_flattened(
                np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8),
                FlattenedOrdering.ROW_MAJOR,
                edge_colors=2,
                is_directed=True,
                allow_loops=True,
            ),
            0.5,
            None,
            FlattenedOrdering.ROW_MAJOR,
            3,
            np.asarray(
                [
                    [0, 0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1, 1, 1, 1],
                ],
                np.uint8,
            ),
        ),
        (
            Graph.from_flattened(
                np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8),
                FlattenedOrdering.ROW_MAJOR,
                edge_colors=2,
                is_directed=True,
                allow_loops=True,
            ),
            1.0,
            None,
            FlattenedOrdering.ROW_MAJOR,
            3,
            np.asarray(
                [
                    [0, 0, 0, 0, 0, 1, 0, 0, 1],
                    [1, 1, 1, 0, 1, 0, 0, 0, 1],
                    [0, 1, 1, 0, 1, 1, 1, 1, 0],
                ],
                np.uint8,
            ),
        ),
        (
            Graph.from_flattened(
                np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8),
                FlattenedOrdering.ROW_MAJOR,
                edge_colors=2,
                is_directed=True,
                allow_loops=True,
            ),
            1.0,
            1.0,
            FlattenedOrdering.ROW_MAJOR,
            3,
            np.asarray(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
                np.uint8,
            ),
        ),
        (
            Graph.from_flattened(
                np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8),
                FlattenedOrdering.ROW_MAJOR,
                edge_colors=2,
                is_directed=True,
                allow_loops=True,
            ),
            1.0,
            0.0,
            FlattenedOrdering.ROW_MAJOR,
            3,
            np.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                np.uint8,
            ),
        ),
        (
            Graph.from_flattened(
                np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.uint8),
                FlattenedOrdering.ROW_MAJOR,
                edge_colors=3,
                is_directed=True,
                allow_loops=True,
            ),
            0.5,
            None,
            FlattenedOrdering.ROW_MAJOR,
            3,
            np.asarray(
                [
                    [0, 0, 0, 1, 0, 1, 2, 2, 1],
                    [1, 0, 0, 1, 1, 2, 0, 2, 0],
                    [0, 0, 0, 1, 1, 1, 2, 2, 2],
                ],
                np.uint8,
            ),
        ),
        (
            Graph.from_flattened(
                np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.uint8),
                FlattenedOrdering.ROW_MAJOR,
                edge_colors=3,
                is_directed=True,
                allow_loops=True,
            ),
            1.0,
            np.asarray([0.0, 0.0, 1.0]),
            FlattenedOrdering.ROW_MAJOR,
            3,
            np.asarray(
                [
                    [2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2],
                ],
                np.uint8,
            ),
        ),
    ],
)
def test_edge_pertrubation(
    graph: Graph,
    edge_prob: float,
    color_prob: np.ndarray | float | None,
    ordering: FlattenedOrdering,
    batch_size: int,
    result: np.ndarray,
):
    rng = np.random.default_rng(42)

    out = create_edge_perturbation_graph_generator(
        initial_graph=graph,
        edge_perturbation_probability=edge_prob,
        color_selection_probabilities=color_prob,
        flattened_ordering=ordering,
        random_generator=rng,
    )(batch_size)

    np.testing.assert_array_equal(out.flattened_row_major_colors, result)


@pytest.mark.parametrize(
    "order, color_prob, ordering, edge_colors, is_directed, allow_loops, batch_size, result",
    [
        (
            2,
            None,
            FlattenedOrdering.ROW_MAJOR,
            2,
            False,
            False,
            3,
            np.asarray([[1], [0], [1]], np.uint8),
        ),
        (
            2,
            1.0,
            FlattenedOrdering.ROW_MAJOR,
            2,
            False,
            False,
            3,
            np.asarray([[1], [1], [1]], np.uint8),
        ),
        (
            2,
            None,
            FlattenedOrdering.ROW_MAJOR,
            2,
            True,
            False,
            3,
            np.asarray([[1, 0], [1, 0], [1, 1]], np.uint8),
        ),
        (
            2,
            None,
            FlattenedOrdering.ROW_MAJOR,
            2,
            False,
            True,
            3,
            np.asarray([[1, 0, 1], [0, 1, 1], [0, 1, 1]], np.uint8),
        ),
        (
            2,
            None,
            FlattenedOrdering.ROW_MAJOR,
            2,
            True,
            True,
            3,
            np.asarray([[1, 0, 1, 0], [1, 1, 0, 1], [1, 1, 1, 1]], np.uint8),
        ),
        (
            2,
            None,
            FlattenedOrdering.CLOCKWISE,
            2,
            True,
            True,
            3,
            np.asarray([[1, 0, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]], np.uint8),
        ),
        (
            3,
            None,
            FlattenedOrdering.ROW_MAJOR,
            2,
            False,
            False,
            3,
            np.asarray([[1, 0, 1], [0, 1, 1], [0, 1, 1]], np.uint8),
        ),
        (
            3,
            None,
            FlattenedOrdering.ROW_MAJOR,
            4,
            False,
            False,
            3,
            np.asarray([[2, 0, 3], [0, 3, 3], [0, 3, 3]], np.uint8),
        ),
        (
            3,
            np.asarray([0.0, 0.5, 0.0, 0.5]),
            FlattenedOrdering.ROW_MAJOR,
            4,
            False,
            False,
            3,
            np.asarray([[3, 1, 3], [3, 1, 3], [3, 3, 1]], np.uint8),
        ),
    ],
)
def test_random_graph(
    order: int,
    color_prob: np.ndarray | float | None,
    ordering: FlattenedOrdering,
    edge_colors: int,
    is_directed: bool,
    allow_loops: bool,
    batch_size: int,
    result: np.ndarray,
):
    rng = np.random.default_rng(42)

    out = create_random_graph_generator(
        graph_order=order,
        color_selection_probabilities=color_prob,
        flattened_ordering=ordering,
        edge_colors=edge_colors,
        is_directed=is_directed,
        allow_loops=allow_loops,
        random_generator=rng,
    )(batch_size)

    np.testing.assert_array_equal(out.flattened_row_major_colors, result)
