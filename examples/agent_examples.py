import numpy as np
import torch.nn as nn
import torch.optim as optim

from rlgt.agents import DeepCrossEntropyAgent
from rlgt.environments import LinearBuildEnvironment
from rlgt.graphs import Graph


def graph_invariant(graph_batch: Graph):
    # Extract the adjacency matrices.
    adjacency_matrix_batch = graph_batch.adjacency_matrix_colors.astype(np.float64)

    # Compute the vertex degrees.
    d_batch = adjacency_matrix_batch.sum(axis=2)
    d_batch_fixed = np.maximum(d_batch, 1)

    # Compute the means of the neighbor degrees.
    m_batch = adjacency_matrix_batch @ d_batch[..., None]
    m_batch = m_batch[..., 0] / d_batch_fixed
    m_batch_fixed = np.maximum(m_batch, 1)

    # Compute the Laplacian matrices.
    laplacian_matrix_batch = -adjacency_matrix_batch
    index_range = np.arange(adjacency_matrix_batch.shape[1])
    laplacian_matrix_batch[:, index_range, index_range] += d_batch

    # Compute the Laplacian spectral radii.
    spectrum_batch = np.linalg.eigvalsh(laplacian_matrix_batch)
    mu_batch = spectrum_batch[:, -1]

    # Compute the difference between the left-hand side and the right-hand side.
    right_hand_side_batch = np.max(m_batch_fixed**2 / d_batch_fixed + m_batch_fixed, axis=1)
    result = mu_batch - right_hand_side_batch

    # Determine whether each of the graphs is connected or disconnected.
    temp = graph_batch.adjacency_matrix_colors.astype(bool) | np.eye(
        graph_batch.graph_order, dtype=bool
    )
    power = 1
    while power < graph_batch.graph_order - 1:
        temp = (temp @ temp).astype(bool)
        power *= 2

    # Punish the disconnected graphs.
    result[~np.all(temp[:, 0, :], axis=1)] = -10.0

    return result


def a1_example(graph_order: int):
    policy_network = nn.Sequential(
        nn.Linear(graph_order * (graph_order - 1), 72),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(72, 12),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(12, 2),
    )

    agent = DeepCrossEntropyAgent(
        environment=LinearBuildEnvironment(
            graph_invariant=graph_invariant,
            graph_order=graph_order,
        ),
        policy_network=policy_network,
        optimizer=optim.Adam(policy_network.parameters(), lr=0.003),
    )

    print("Deep Cross-Entropy agent + Linear Build environment")
    agent.reset()

    while True:
        agent.step()
        print(f"Learning iterations: {agent.step_count}. Best score: {agent.best_score:.3f}.")

        if agent.best_score > 0.0001:
            print("Success! The following graph is a solution:")
            solution = agent.best_graph.adjacency_matrix_colors
            print(solution)

            break


if __name__ == "__main__":
    a1_example(graph_order=16)
