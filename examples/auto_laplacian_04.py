import numpy as np
import torch.nn as nn
import torch.optim as optim

from rl_graph_theory.graphs.graph import Graph
from rl_graph_theory.graphs.graph_formats import FlattenedOrdering
from rl_graph_theory.agents.deep_cross_entropy_method import DeepCrossEntropyMethod
from rl_graph_theory.environments.linear_environments import LinearBuildEnvironment, LinearFlipEnvironment
from rl_graph_theory.environments.local_environments import LocalSetEnvironment
from rl_graph_theory.environments.global_environments import GlobalSetEnvironment
from rl_graph_theory.environments.graph_environment import RewardType
from rl_graph_theory.agents.random_action_mechanisms import create_multiplication_factor_random_action_mechanism


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

    temp = np.max((2.0 * degree_batch_1 ** 2) / and_batch_1, axis=1)
    result = mu_batch - temp

    result[spectrum_batch[:, 1] < 0.15] = -1000.0
    result[np.min(degree_batch, axis=1) < 0.5] = -1000.0
    result = result.astype(np.float32)

    return result


def main(graph_order: int):
    policy_network = nn.Sequential(
        nn.Linear(graph_order * (graph_order - 1), 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2),
    )

    dcem = DeepCrossEntropyMethod(
        environment=LinearBuildEnvironment(
            reward_type=RewardType.SPARSE,
            reward_function=graph_invariant,
            graph_order=graph_order,
        ),
        policy_network=policy_network,
        optimizer=optim.Adam(policy_network.parameters(), lr=0.003),
        loss_function=nn.CrossEntropyLoss(),
        new_candidates_count=500,
        elite_count=30,
        survivors_count=10,
        random_action_mechanism=create_multiplication_factor_random_action_mechanism(
            initial_random_action_probability=0.005,
            waiting_period=10,
            multiplication_factor=1.1,
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
            with open(f"examples/auto_laplacian_01_result_{graph_order}.txt", "w") as opened_file:
                opened_file.write(np.array2string(solution, separator=", "))

            break


if __name__ == "__main__":
    main(graph_order=16)