import numpy as np
import torch.nn as nn
import torch.optim as optim

from rlgt.agents import DeepCrossEntropyAgent
from rlgt.environments import LinearBuildEnvironment
from rlgt.graphs import Graph


def graph_invariant(graph_batch: Graph):
    adj_batch = graph_batch.adjacency_matrix_colors.astype(np.float64)

    degree_batch = adj_batch.sum(axis=2)
    degree_batch_1 = np.maximum(degree_batch, 1)

    and_batch = adj_batch @ degree_batch[..., None]
    and_batch = and_batch[..., 0] / degree_batch_1
    and_batch_1 = np.maximum(and_batch, 1)

    lap_batch = -adj_batch
    i = np.arange(adj_batch.shape[1])
    lap_batch[:, i, i] += degree_batch

    spectrum_batch = np.linalg.eigvalsh(lap_batch)
    mu_batch = spectrum_batch[:, -1]

    temp = np.max(
        and_batch_1 ** 2 / degree_batch_1 + and_batch_1, axis=1
    )
    result = mu_batch - temp

    result[spectrum_batch[:, 1] < 0.15] = -5.0
    result[np.min(degree_batch, axis=1) < 0.5] = -5.0
    result = result.astype(np.float32)

    return result


def agent_example(graph_order: int):
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

    agent.reset()

    while True:
        agent.step()
        print(f"Generations: {agent.step_count}. Best score: {agent.best_score:.3f}.")

        if agent.best_score > 0.0001:
            print("Success!")
            solution = agent.best_graph.adjacency_matrix_colors
            print(solution)

            break


if __name__ == "__main__":
    agent_example(graph_order=16)