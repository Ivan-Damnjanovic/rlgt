import numpy as np
import torch.nn as nn
import torch.optim as optim

from rlgt.agents import ExponentialRandomActionMechanism, ReinforceAgent
from rlgt.environments import LocalSetEnvironment, create_fixed_graph_generator, GlobalFlipEnvironment
from rlgt.graphs import CycleGraph, Graph, GraphFormat


AUTO_LAPLACIAN_EXPRESSIONS = {
    3: lambda d, m: (m**2) / d + m,
    15: lambda d, m: np.sqrt(4 * m**3 / d),
    26: lambda d, m: np.sqrt(5 * d**2 + 11 * d * m) / 2,
    28: lambda d, m: np.sqrt((2 * m**4) / (d**2) + 2 * d * m),
    29: lambda d, m: np.sqrt(m**2 + (3 * m**3) / d),
    31: lambda d, m: (4 * m**2) / (m + d),
}


def graph_invariant(graph_batch: Graph, expression_index: int):
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
        AUTO_LAPLACIAN_EXPRESSIONS[expression_index](d=degree_batch_1, m=and_batch_1), axis=1
    )
    result = mu_batch - temp

    result[spectrum_batch[:, 1] < 0.15] = -5.0
    result[np.min(degree_batch, axis=1) < 0.5] = -5.0
    result = result.astype(np.float32)

    return result


def main(graph_order: int, expression_index: int):
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
            graph_invariant=lambda graph_batch: graph_invariant(
                graph_batch=graph_batch, expression_index=expression_index
            ),
            graph_order=graph_order,
            flip_only=True,
            initial_graph_generator=create_fixed_graph_generator(
                fixed_graph=CycleGraph(
                    graph_formats={GraphFormat.FLATTENED_ROW_MAJOR_COLORS},
                    graph_order=graph_order,
                ),
                graph_format=GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
            ),
            episode_length=30,
        ),
        policy_network=policy_network,
        optimizer=optim.Adam(policy_network.parameters(), lr=0.001),
        candidates_count=200,
        random_action_mechanism=ExponentialRandomActionMechanism(
            initial_random_action_probability=0.005,
            waiting_period=10,
            multiplicative_factor=1.1,
            maximum_random_action_probability=0.100,
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
            with open(
                "applications/auto_laplacian_results/"
                + f"reinforce_{expression_index:02}_{graph_order}.txt",
                "w",
            ) as opened_file:
                opened_file.write(np.array2string(solution, separator=", "))

            break


if __name__ == "__main__":
    main(graph_order=16, expression_index=3)
    print()
    main(graph_order=16, expression_index=15)
    print()
    main(graph_order=16, expression_index=28)
    print()
    main(graph_order=16, expression_index=29)
    print()
    main(graph_order=16, expression_index=31)
