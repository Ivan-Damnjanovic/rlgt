import numpy as np
import torch.nn as nn
import torch.optim as optim

from rlgt.graphs import Graph
from rlgt.agents import DeepCrossEntropyAgent, ExponentialRandomActionMechanism, ReinforceAgent
from rlgt.environments import LinearBuildEnvironment, LinearSetEnvironment, LinearFlipEnvironment


def auto_laplacian_expression(d, m):
    return (4 * m ** 2) / (m + d)


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

    temp = np.max(auto_laplacian_expression(d=degree_batch_1, m=and_batch_1), axis=1)
    result = mu_batch - temp

    result[spectrum_batch[:, 1] < 0.15] = -1000.0
    result[np.min(degree_batch, axis=1) < 0.5] = -1000.0
    result = result.astype(np.float32)

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

    dcem = ReinforceAgent(
        environment=LinearBuildEnvironment(
            graph_invariant=graph_invariant,
            graph_order=graph_order,
        ),
        policy_network=policy_network,
        candidates_count=200,
        elite_count=30,
        optimizer=optim.Adam(policy_network.parameters(), lr=0.003),
        random_action_mechanism=ExponentialRandomActionMechanism(
            initial_random_action_probability=0.005,
            waiting_period=10,
            multiplicative_factor=1.1,
            maximum_random_action_probability=0.025,
        ),
    )

    # dcem = DeepCrossEntropyAgent(
    #     environment=LinearBuildEnvironment(
    #         graph_invariant=graph_invariant,
    #         graph_order=graph_order,
    #     ),
    #     policy_network=policy_network,
    #     optimizer=optim.Adam(policy_network.parameters(), lr=0.003),
    #     # random_action_mechanism=ExponentialRandomActionMechanism(
    #     #     initial_random_action_probability=0.005,
    #     #     waiting_period=10,
    #     #     multiplicative_factor=1.1,
    #     #     maximum_random_action_probability=0.025,
    #     # ),
    # )


    dcem.reset()

    while True:
        dcem.step()
        print(f"Generations: {dcem.step_count}. Best score: {dcem.best_score:.3f}.")

        if dcem.best_score > 0.0001:
            print("Success!")
            solution = dcem.best_graph.adjacency_matrix_colors
            
            print(solution)
            with open(f"applications/auto_laplacian_31_result_{graph_order}.txt", "w") as opened_file:
                opened_file.write(np.array2string(solution, separator=", "))

            break


if __name__ == "__main__":
    main(graph_order=14)