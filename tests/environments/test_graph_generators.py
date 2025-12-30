import pytest
import numpy as np

from rl_graph_theory.graphs.graph import Graph, GraphFormat, FlattenedOrdering
from rl_graph_theory.environments.graph_generators import (
    create_fixed_graph_generator,
    create_choose_two_graph_generator,
)


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
        np.asarray([1, 2, 1], dtype=np.uint8),
        FlattenedOrdering.ROW_MAJOR,
        edge_colors=3,
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
]


def combinations(a, b):
    for _a in a:
        for _b in b:
            yield (_a, _b)


def verify_generator(generator, graphs):
    gb = generator(len(graphs))
    np.testing.assert_array_equal(
        gb.flattened_row_major, np.stack([g.flattened_row_major for g in graphs], axis=0)
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
        first_graph.flattened_row_major,
        FlattenedOrdering.ROW_MAJOR,
        first_graph.edge_colors,
        first_graph.is_directed,
        first_graph.allow_loops,
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
