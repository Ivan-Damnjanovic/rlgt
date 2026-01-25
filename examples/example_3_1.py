import numpy as np

from rl_graph_theory.graphs import (
    BitmaskType,
    ColorRepresentation,
    FlattenedOrdering,
    Graph,
)


def g1_example():
    flattened_row_major_colors = np.array(
        [2, 2, 2, 0, 3, 3, 3, 0, 3, 2, 1, 2, 1, 2, 3, 1],
        dtype=np.uint8,
    )
    g1 = Graph(
        edge_colors=3,
        is_directed=True,
        allow_loops=True,
        flattened_row_major_colors=flattened_row_major_colors,
    )

    print("Graph G_1")
    print(g1.bitmask_out)
    print(g1.bitmask_in)
    print(g1.adjacency_matrix_colors)
    print(g1.adjacency_matrix_binary)
    print(g1.flattened_row_major_colors)
    print(g1.flattened_row_major_binary)
    print(g1.flattened_clockwise_colors)
    print(g1.flattened_clockwise_binary)


def g2_example():
    bitmask = np.array(
        [
            [8, 12, 0, 0],
            [6, 0, 9, 2],
        ],
        dtype=np.uint64,
    )
    g2 = Graph.from_bitmask(
        bitmask=bitmask,
        bitmask_type=BitmaskType.OUT_NEIGHBORS,
        edge_colors=3,
        is_directed=True,
        allow_loops=False,
    )

    print("Graph G_2")
    print(g2.bitmask_out)
    print(g2.bitmask_in)
    print(g2.adjacency_matrix_colors)
    print(g2.adjacency_matrix_binary)
    print(g2.flattened_row_major_colors)
    print(g2.flattened_row_major_binary)
    print(g2.flattened_clockwise_colors)
    print(g2.flattened_clockwise_binary)


def g3_example():
    adjacency_matrix = np.array(
        [
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ],
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ],
            [
                [0, 0, 1],
                [0, 0, 0],
                [1, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
        ],
        dtype=np.uint8,
    )
    g3 = Graph.from_adjacency_matrix(
        adjacency_matrix=adjacency_matrix,
        color_representation=ColorRepresentation.BINARY_SLICES,
        edge_colors=4,
        is_directed=False,
        allow_loops=True,
    )

    print("Graph G_3")
    print(g3.bitmask_out)
    print(g3.bitmask_in)
    print(g3.adjacency_matrix_colors)
    print(g3.adjacency_matrix_binary)
    print(g3.flattened_row_major_colors)
    print(g3.flattened_row_major_binary)
    print(g3.flattened_clockwise_colors)
    print(g3.flattened_clockwise_binary)


def g4_example():
    flattened = np.array([0, 1, 0, 1, 1, 1, 0, 1, 1, 1], dtype=np.uint8)
    g4 = Graph.from_flattened(
        flattened=flattened,
        flattened_ordering=FlattenedOrdering.CLOCKWISE,
        color_representation=ColorRepresentation.COLOR_NUMBERS,
    )

    print("Graph G_4")
    print(g4.bitmask_out)
    print(g4.bitmask_in)
    print(g4.adjacency_matrix_colors)
    print(g4.adjacency_matrix_binary)
    print(g4.flattened_row_major_colors)
    print(g4.flattened_row_major_binary)
    print(g4.flattened_clockwise_colors)
    print(g4.flattened_clockwise_binary)


if __name__ == "__main__":
    g1_example()
    print()
    g2_example()
    print()
    g3_example()
    print()
    g4_example()
