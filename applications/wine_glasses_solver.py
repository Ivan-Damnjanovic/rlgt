import numpy as np
from rlgt import graphs as rlgt_graphs
from rlgt import environments as rlgt_environments
from rlgt import agents as rlgt_agents
import torch.nn as nn
import torch.optim as optim
from sage.all import *


def graph_invariant(graph_batch: rlgt_graphs.Graph):
    result = np.zeros(graph_batch.batch_size, dtype=np.float32)

    for index in range(graph_batch.batch_size):
        g = Graph(matrix(graph_batch.adjacency_matrix_colors[index]))

        if not g.is_connected():
            result[index] = -1000.0
            continue

        delta = max(g.degree())

        if delta > 5:
            result[index] = -2000.0
            continue

        mu = len(g.matching())
        eigenvalues = g.adjacency_matrix().eigenvalues()
        energy = sum(abs(eigenvalue) for eigenvalue in eigenvalues)

        result[index] = energy - 2 * mu * sqrt(delta)

    return result


def main(graph_order: int):
    policy_network = nn.Sequential(
        nn.Linear(graph_order * (graph_order - 1), 72),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(72, 12),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(12, 2),
    )

    agent = rlgt_agents.DeepCrossEntropyAgent(
        environment=rlgt_environments.LinearBuildEnvironment(
            graph_invariant=graph_invariant,
            graph_order=graph_order,
        ),
        policy_network=policy_network,
        optimizer=optim.Adam(policy_network.parameters(), lr=0.003),
        random_action_mechanism=rlgt_agents.ExponentialRandomActionMechanism(
            initial_random_action_probability=0.005,
            waiting_period=10,
            multiplicative_factor=1.1,
            maximum_random_action_probability=0.025,
        ),
    )

    agent.reset()

    while True:
        agent.step()
        print(f"Generations: {agent.step_count}. Best score: {agent.best_score:.3f}.")

        if agent.best_score > 0.0001:
            print("Success!")
            solution = agent.best_graph.adjacency_matrix_colors

            print(solution)
            with open(f"applications/wine_glasses_{graph_order}.txt", "w") as opened_file:
                opened_file.write(np.array2string(solution, separator=", "))

            break


if __name__ == "__main__":
    main(graph_order=14)
