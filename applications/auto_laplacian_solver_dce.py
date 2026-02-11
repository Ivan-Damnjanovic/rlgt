import numpy as np
import torch.nn as nn
import torch.optim as optim

from rlgt.agents import DeepCrossEntropyAgent, ExponentialRandomActionMechanism
from rlgt.environments import LinearBuildEnvironment
from rlgt.graphs import Graph


LAPLACIAN_EXPRESSIONS = {
    3: lambda d, m: (m**2) / d + m,
    15: lambda d, m: np.sqrt(4 * m**3 / d),
    26: lambda d, m: np.sqrt(5 * d**2 + 11 * d * m) / 2,
    28: lambda d, m: np.sqrt((2 * m**4) / (d**2) + 2 * d * m),
    29: lambda d, m: np.sqrt(m**2 + (3 * m**3) / d),
    31: lambda d, m: (4 * m**2) / (m + d),
    36: lambda du, mu, dv, mv: 2 * (mu**2 + mv**2) / (du + dv),
}


def graph_invariant_family(graph_batch: Graph, expression_index: int):
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

    if expression_index <= 32:    
        right_hand_side_batch = np.max(
            LAPLACIAN_EXPRESSIONS[expression_index](d_batch_fixed, m_batch_fixed), axis=1
        )
    else:
        b, u, v = np.nonzero(np.triu(graph_batch.adjacency_matrix_colors, k=1))
        
        du = d_batch_fixed[b, u]
        mu = m_batch_fixed[b, u]
        dv = d_batch_fixed[b, v]
        mv = m_batch_fixed[b, v]

        all_right_hand_sides = LAPLACIAN_EXPRESSIONS[expression_index](du, mu, dv, mv)

        right_hand_side_batch = np.full(graph_batch.batch_size, -np.inf)
        np.maximum.at(right_hand_side_batch, b, all_right_hand_sides)

    # Compute the differences between the left-hand side and the right-hand side.
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


def main(graph_order: int, expression_index: int):
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
            graph_invariant=lambda graph_batch: graph_invariant_family(
                graph_batch=graph_batch, expression_index=expression_index
            ),
            graph_order=graph_order,
        ),
        policy_network=policy_network,
        optimizer=optim.Adam(policy_network.parameters(), lr=0.003),
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
                f"applications/auto_laplacian_results/dce_{expression_index:02}_{graph_order}.txt",
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
