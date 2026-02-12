import numpy as np
import torch.nn as nn
import torch.optim as optim

from rlgt.agents import DeepCrossEntropyAgent, PPOAgent, ReinforceAgent
from rlgt.environments import (
    GlobalFlipEnvironment,
    LinearBuildEnvironment,
    LocalSetEnvironment,
    create_fixed_graph_generator,
)
from rlgt.graphs import CycleGraph, Graph, GraphFormat


def graph_invariant(graph_batch: Graph) -> np.ndarray:
    r"""
    This function computes the graph invariant
    \[
        \mu - \max_{v \in V} \left( m(v)^2 / d(v) + m(v) \right)
    \]
    for a provided batch of graphs, where $\mu$ is the Laplacian spectral radius, $d(v)$ is the
    degree of a vertex $v$, $m(v)$ is the average degree of the neighbors of a vertex $v$, and the
    maximum is taken over all the graph vertices $v$. An input graph is assumed to be connected;
    otherwise, a score of -10.0 is returned.

    :param graph_batch: The provided batch of graphs, given as a `Graph` object.

    :return: The computed batch of graph invariant values, given as a `numpy.ndarray` of type
        `numpy.float32`.
    """

    # Extract the adjacency matrices.
    adjacency_matrix_batch = graph_batch.adjacency_matrix_colors.astype(np.float64)

    # Compute the vertex degrees.
    d_batch = adjacency_matrix_batch.sum(axis=2)
    d_batch_fixed = np.maximum(d_batch, 1)

    # Compute the average degrees of the vertex neighbors.
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

    # Compute the differences between the left-hand side and the right-hand side.
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

    return result.astype(np.float32)


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
    print("Starting...")
    agent.reset()

    while True:
        agent.step()
        print(f"Learning iterations: {agent.step_count}. Best score: {agent.best_score:.3f}.")

        if agent.best_score > 0.0001:
            print("Success! The following graph is a solution:")
            print(agent.best_graph.adjacency_matrix_colors)

            break

        if agent.step_count >= 1000:
            print("Restarting...")
            agent.reset()


def a2_example(graph_order: int):
    policy_network = nn.Sequential(
        nn.Linear(graph_order * (graph_order - 1) // 2, 72),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(72, 12),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(12, graph_order * (graph_order - 1)),
    )

    agent = ReinforceAgent(
        environment=GlobalFlipEnvironment(
            graph_invariant=graph_invariant,
            graph_order=graph_order,
            episode_length=30,
            flip_only=True,
            initial_graph_generator=create_fixed_graph_generator(
                fixed_graph=CycleGraph(
                    graph_formats={GraphFormat.FLATTENED_ROW_MAJOR_COLORS},
                    graph_order=graph_order,
                ),
                graph_format=GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
            ),
        ),
        policy_network=policy_network,
        optimizer=optim.Adam(policy_network.parameters(), lr=0.001),
    )

    print("REINFORCE agent + Global Flip environment")
    print("Starting...")
    agent.reset()

    while True:
        agent.step()
        print(f"Learning iterations: {agent.step_count}. Best score: {agent.best_score:.3f}.")

        if agent.best_score > 0.0001:
            print("Success! The following graph is a solution:")
            print(agent.best_graph.adjacency_matrix_colors)

            break

        if agent.step_count >= 200:
            print("Restarting...")
            agent.reset()


def a3_example(graph_order: int):
    policy_network = nn.Sequential(
        nn.Linear(graph_order * (graph_order - 1) // 2 + graph_order, 72),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(72, 12),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(12, graph_order * 2),
    )

    value_network = nn.Sequential(
        nn.Linear(graph_order * (graph_order - 1) // 2 + graph_order, 72),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(72, 12),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(12, 1),
    )

    agent = PPOAgent(
        environment=LocalSetEnvironment(
            graph_invariant=graph_invariant,
            graph_order=graph_order,
            episode_length=30,
            initial_graph_generator=create_fixed_graph_generator(
                fixed_graph=CycleGraph(
                    graph_formats={GraphFormat.FLATTENED_ROW_MAJOR_COLORS},
                    graph_order=graph_order,
                ),
                graph_format=GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
            ),
        ),
        policy_network=policy_network,
        value_network=value_network,
        optimizer=optim.Adam(
            list(policy_network.parameters()) + list(value_network.parameters()), lr=0.001
        ),
    )

    print("PPO agent + Local Set environment")
    print("Starting...")
    agent.reset()

    while True:
        agent.step()
        print(f"Learning iterations: {agent.step_count}. Best score: {agent.best_score:.3f}.")

        if agent.best_score > 0.0001:
            print("Success! The following graph is a solution:")
            print(agent.best_graph.adjacency_matrix_colors)

            break

        if agent.step_count >= 200:
            print("Restarting...")
            agent.reset()


if __name__ == "__main__":
    a1_example(graph_order=16)
    print()
    a2_example(graph_order=16)
    print()
    a3_example(graph_order=16)
