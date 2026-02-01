import numpy as np

from rl_graph_theory.environments import (
    GlobalFlipEnvironment,
    LinearBuildEnvironment,
    RewardType,
    create_fixed_graph_generator,
)
from rl_graph_theory.graphs import (
    FlattenedOrdering,
    Graph,
    GraphFormat,
    MonochromaticGraph,
)


def e1_example():
    def reward_function(graph_batch: Graph):
        zero_color_mask = (graph_batch.flattened_row_major_colors == 0).astype(np.float32)
        return np.sum(zero_color_mask, axis=1) ** 2

    e1 = LinearBuildEnvironment(
        reward_type=RewardType.TELESCOPIC,
        reward_function=reward_function,
        graph_order=3,
        flattened_ordering=FlattenedOrdering.CLOCKWISE,
        edge_colors=4,
        allow_loops=True,
    )

    print("Linear Build environment")
    state_batch, status = e1.reset_batch(4)
    print("Step 0")
    print(state_batch)
    print(status)
    print(e1.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, reward_batch, status = e1.step_batch(np.array([0, 0, 0, 1], dtype=np.int32))
    print("Step 1")
    print(state_batch)
    print(reward_batch)
    print(status)
    print(e1.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, reward_batch, status = e1.step_batch(np.array([3, 2, 1, 3], dtype=np.int32))
    print("Step 2")
    print(state_batch)
    print(reward_batch)
    print(status)
    print(e1.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, reward_batch, status = e1.step_batch(np.array([0, 3, 0, 1], dtype=np.int32))
    print("Step 3")
    print(state_batch)
    print(reward_batch)
    print(status)
    print(e1.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, reward_batch, status = e1.step_batch(np.array([1, 0, 2, 2], dtype=np.int32))
    print("Step 4")
    print(state_batch)
    print(reward_batch)
    print(status)
    print(e1.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, reward_batch, status = e1.step_batch(np.array([1, 2, 3, 0], dtype=np.int32))
    print("Step 5")
    print(state_batch)
    print(reward_batch)
    print(status)
    print(e1.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, reward_batch, status = e1.step_batch(np.array([2, 0, 0, 1], dtype=np.int32))
    print("Step 6")
    print(state_batch)
    print(reward_batch)
    print(status)
    print(e1.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)


def e2_example():
    def reward_function(graph_batch: Graph):
        degrees = np.sum(graph_batch.adjacency_matrix_colors, axis=2)
        return np.sum(degrees**2, axis=1).astype(np.float32)

    e2 = GlobalFlipEnvironment(
        reward_type=RewardType.SPARSE,
        reward_function=reward_function,
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
    state_batch, status = e2.reset_batch(2)
    print("Step 0")
    print(state_batch)
    print(status)
    print(e2.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, reward_batch, status = e2.step_batch(np.array([0, 2], dtype=np.int32))
    print("Step 1")
    print(state_batch)
    print(reward_batch)
    print(status)
    print(e2.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, reward_batch, status = e2.step_batch(np.array([1, 7], dtype=np.int32))
    print("Step 2")
    print(state_batch)
    print(reward_batch)
    print(status)
    print(e2.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, reward_batch, status = e2.step_batch(np.array([5, 1], dtype=np.int32))
    print("Step 3")
    print(state_batch)
    print(reward_batch)
    print(status)
    print(e2.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)

    state_batch, reward_batch, status = e2.step_batch(np.array([9, 7], dtype=np.int32))
    print("Step 4")
    print(state_batch)
    print(reward_batch)
    print(status)
    print(e2.state_batch_to_graph_batch(state_batch).adjacency_matrix_colors)


if __name__ == "__main__":
    e1_example()
    print()
    e2_example()
    print()
