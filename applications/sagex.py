from sage.all import *
import rl_graph_theory.graphs as rl_g
import rl_graph_theory.environments as rl_e
import rl_graph_theory.agents as rl_a
import numpy as np
import torch.nn as nn
import torch.optim as optim


def graph_invariant(graph_batch: rl_g.Graph):
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

    dcem = rl_a.DeepCrossEntropyAgent(
        environment=rl_e.LinearBuildEnvironment(
            reward_type=rl_e.RewardType.SPARSE,
            reward_function=graph_invariant,
            graph_order=graph_order,
        ),
        policy_network=policy_network,
        optimizer=optim.Adam(policy_network.parameters(), lr=0.003),
        loss_function=nn.CrossEntropyLoss(),
        candidates_count=200,
        elite_count=20,
        survivors_count=5,
        random_action_mechanism=rl_a.ExponentialRandomActionMechanism(
            initial_random_action_probability=0.005,
            waiting_period=10,
            multiplicative_factor=1.1,
            maximum_random_action_probability=0.025,
        ),
    )

    dcem.reset()

    while True:
        dcem.step()
        print(f"Generations: {dcem.step_count}. Best score: {dcem.best_score:.3f}.")

        if dcem.best_score > 0.0001:
            print("Success!")
            solution = dcem.best_graph.adjacency_matrix_colors
            
            print(solution)
            with open(f"examples/auto_laplacian_31_result_{graph_order}.txt", "w") as opened_file:
                opened_file.write(np.array2string(solution, separator=", "))

            break


if __name__ == "__main__":
    main(graph_order=14)