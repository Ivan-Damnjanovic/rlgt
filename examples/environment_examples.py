import numpy as np

from rlgt.environments import (
    GlobalFlipEnvironment,
    LinearBuildEnvironment,
    LocalSetEnvironment,
    create_fixed_graph_generator,
)
from rlgt.graphs import FlattenedOrdering, Graph, GraphFormat, MonochromaticGraph


def e1_example():
    def graph_invariant(graph_batch: Graph):
        zero_color_mask = (graph_batch.flattened_row_major_colors == 0).astype(np.float32)
        return np.sum(zero_color_mask, axis=1) ** 2

    e1 = LinearBuildEnvironment(
        graph_invariant=graph_invariant,
        graph_order=3,
        flattened_ordering=FlattenedOrdering.CLOCKWISE,
        edge_colors=4,
        allow_loops=True,
    )

    print("Linear Build environment")
    state_batch, graph_invariant_batch, status = e1.reset_batch(4)
    print("Step 0")
    print(state_batch)
    print(graph_invariant_batch)
    print(status)
    print(e1.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, graph_invariant_batch, status = e1.step_batch(
        np.array([0, 0, 0, 1], dtype=np.int32)
    )
    print("Step 1")
    print(state_batch)
    print(graph_invariant_batch)
    print(status)
    print(e1.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, graph_invariant_batch, status = e1.step_batch(
        np.array([3, 2, 1, 3], dtype=np.int32)
    )
    print("Step 2")
    print(state_batch)
    print(graph_invariant_batch)
    print(status)
    print(e1.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, graph_invariant_batch, status = e1.step_batch(
        np.array([0, 3, 0, 1], dtype=np.int32)
    )
    print("Step 3")
    print(state_batch)
    print(graph_invariant_batch)
    print(status)
    print(e1.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, graph_invariant_batch, status = e1.step_batch(
        np.array([1, 0, 2, 2], dtype=np.int32)
    )
    print("Step 4")
    print(state_batch)
    print(graph_invariant_batch)
    print(status)
    print(e1.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, graph_invariant_batch, status = e1.step_batch(
        np.array([1, 2, 3, 0], dtype=np.int32)
    )
    print("Step 5")
    print(state_batch)
    print(graph_invariant_batch)
    print(status)
    print(e1.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, graph_invariant_batch, status = e1.step_batch(
        np.array([2, 0, 0, 1], dtype=np.int32)
    )
    print("Step 6")
    print(state_batch)
    print(graph_invariant_batch)
    print(status)
    print(e1.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)


def e2_example():
    def graph_invariant(graph_batch: Graph):
        degrees = np.sum(graph_batch.adjacency_matrix_colors, axis=2)
        return np.sum(degrees**2, axis=1).astype(np.float32)

    e2 = GlobalFlipEnvironment(
        graph_invariant=graph_invariant,
        graph_order=5,
        episode_length=4,
        flip_only=True,
        flattened_ordering=FlattenedOrdering.ROW_MAJOR,
        initial_graph_generator=create_fixed_graph_generator(
            fixed_graph=MonochromaticGraph(
                graph_formats={GraphFormat.FLATTENED_ROW_MAJOR_COLORS},
                graph_order=5,
                selected_color=1,
            ),
            graph_format=GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
        ),
    )

    print("Global Flip environment")
    state_batch, graph_invariant_batch, status = e2.reset_batch(2)
    print("Step 0")
    print(state_batch)
    print(graph_invariant_batch)
    print(status)
    print(e2.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, graph_invariant_batch, status = e2.step_batch(np.array([0, 2], dtype=np.int32))
    print("Step 1")
    print(state_batch)
    print(graph_invariant_batch)
    print(status)
    print(e2.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, graph_invariant_batch, status = e2.step_batch(np.array([1, 7], dtype=np.int32))
    print("Step 2")
    print(state_batch)
    print(graph_invariant_batch)
    print(status)
    print(e2.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, graph_invariant_batch, status = e2.step_batch(np.array([5, 1], dtype=np.int32))
    print("Step 3")
    print(state_batch)
    print(graph_invariant_batch)
    print(status)
    print(e2.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, graph_invariant_batch, status = e2.step_batch(np.array([9, 7], dtype=np.int32))
    print("Step 4")
    print(state_batch)
    print(graph_invariant_batch)
    print(status)
    print(e2.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)


def e3_example():
    def graph_invariant(graph_batch: Graph):
        adj_1 = graph_batch.adjacency_matrix_binary[:, -2, :, :]
        trace_sum_1 = np.trace(adj_1 @ adj_1 @ adj_1, axis1=1, axis2=2)

        adj_2 = graph_batch.adjacency_matrix_binary[:, -1, :, :]
        trace_sum_2 = np.trace(adj_2 @ adj_2 @ adj_2, axis1=1, axis2=2)

        return (trace_sum_1 + trace_sum_2).astype(np.float32) / 3.0

    e3 = LocalSetEnvironment(
        graph_invariant=graph_invariant,
        graph_order=4,
        episode_length=6,
        flattened_ordering=FlattenedOrdering.ROW_MAJOR,
        edge_colors=3,
        is_directed=True,
        starting_vertex=0,
    )

    print("Local Set environment")
    state_batch, graph_invariant_batch, status = e3.reset_batch(1)
    print("Step 0")
    print(state_batch)
    print(graph_invariant_batch)
    print(status)
    print(e3.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, graph_invariant_batch, status = e3.step_batch(np.array([6], dtype=np.int32))
    print("Step 1")
    print(state_batch)
    print(graph_invariant_batch)
    print(status)
    print(e3.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, graph_invariant_batch, status = e3.step_batch(np.array([7], dtype=np.int32))
    print("Step 2")
    print(state_batch)
    print(graph_invariant_batch)
    print(status)
    print(e3.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, graph_invariant_batch, status = e3.step_batch(np.array([4], dtype=np.int32))
    print("Step 3")
    print(state_batch)
    print(graph_invariant_batch)
    print(status)
    print(e3.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, graph_invariant_batch, status = e3.step_batch(np.array([5], dtype=np.int32))
    print("Step 4")
    print(state_batch)
    print(graph_invariant_batch)
    print(status)
    print(e3.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, graph_invariant_batch, status = e3.step_batch(np.array([7], dtype=np.int32))
    print("Step 5")
    print(state_batch)
    print(graph_invariant_batch)
    print(status)
    print(e3.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, graph_invariant_batch, status = e3.step_batch(np.array([8], dtype=np.int32))
    print("Step 6")
    print(state_batch)
    print(graph_invariant_batch)
    print(status)
    print(e3.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)


if __name__ == "__main__":
    e1_example()
    print()
    e2_example()
    print()
    e3_example()
