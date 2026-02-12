import numpy as np
import torch.nn as nn
import torch.optim as optim

from rlgt.agents import DeepCrossEntropyAgent, ExponentialRandomActionMechanism
from rlgt.environments import LinearBuildEnvironment
from rlgt.graphs import Graph


LAPLACIAN_EXPRESSIONS = {
    1:  lambda d, m: np.sqrt(4 * d**3 / m),
    2:  lambda d, m: 2 * m**2 / d,
    3:  lambda d, m: m**2 / d + m,
    4:  lambda d, m: 2 * d**2 / m,
    5:  lambda d, m: d**2 / m + m,
    6:  lambda d, m: np.sqrt(m**2 + 3 * d**2),
    7:  lambda d, m: d**2 / m + d,
    8:  lambda d, m: np.sqrt(d * (m + 3 * d)),
    9:  lambda d, m: (m + 3 * d) / 2,
    10: lambda d, m: np.sqrt(d * (d + 3 * m)),
    11: lambda d, m: 2 * m**3 / d**2,
    12: lambda d, m: np.sqrt(2 * m**2 + 2 * d**2),
    13: lambda d, m: 2 * m**4 / d**3,
    14: lambda d, m: 2 * d**3 / m**2,
    15: lambda d, m: np.sqrt(4 * m**3 / d),
    16: lambda d, m: 2 * d**4 / m**3,
    17: lambda d, m: (5 * d**4 + 11 * m**4)**0.25,
    18: lambda d, m: np.sqrt(2 * m**3 / d + 2 * d**2),
    19: lambda d, m: (4 * d**4 + 12 * d * m**3)**0.25,
    20: lambda d, m: np.sqrt(7 * d**2 + 9 * m**2) / 2,
    21: lambda d, m: np.sqrt(d**3 / m + 3 * m**2),
    22: lambda d, m: (2 * d**4 + 14 * d**2 * m**2)**0.25,
    23: lambda d, m: np.sqrt(d**2 + 3 * d * m),
    24: lambda d, m: (6 * d**4 + 10 * m**4)**0.25,
    25: lambda d, m: (3 * d**4 + 13 * d**2 * m**2)**0.25,
    26: lambda d, m: np.sqrt(5 * d**2 + 11 * d * m) / 2,
    27: lambda d, m: np.sqrt(3 * d**2 + 5 * d * m) / 2,
    28: lambda d, m: np.sqrt(2 * m**4 / d**2 + 2 * d * m),
    29: lambda d, m: np.sqrt(m**2 + 3 * m**3 / d),
    30: lambda d, m: m**3 / d**2 + d**2 / m,
    31: lambda d, m: 4 * m**2 / (m + d),
    32: lambda d, m: np.sqrt(m**3 * (m + 3 * d)) / d,
    33: lambda du, mu, dv, mv: 2 * (du + dv) - (mu + mv),
    34: lambda du, mu, dv, mv: 2 * (du**2 + dv**2) / (du + dv),
    35: lambda du, mu, dv, mv: 2 * (du**2 + dv**2) / (mu + mv),
    36: lambda du, mu, dv, mv: 2 * (mu**2 + mv**2) / (du + dv),
    37: lambda du, mu, dv, mv: np.sqrt(2 * (du**2 + dv**2)),
    38: lambda du, mu, dv, mv: 2 + np.sqrt(2 * (du - 1)**2 + 2 * (dv - 1)**2),
    39: lambda du, mu, dv, mv: 2 + np.sqrt(2 * (du**2 + dv**2) - 4 * (mu + mv) + 4),
    40: lambda du, mu, dv, mv: 2 + np.sqrt(2 * ((mu - 1)**2 + (mv - 1)**2) + (du**2 + dv**2) - (du * mu + dv * mv)),
    41: lambda du, mu, dv, mv: 2 + (mu + mv) - (du + dv) + np.sqrt(2 * (du**2 + dv**2) - 4 * (mu + mv) + 4),
    42: lambda du, mu, dv, mv: np.sqrt(du**2 + dv**2 + 2 * mu * mv),
    43: lambda du, mu, dv, mv: 2 + np.sqrt(3 * (mu**2 + mv**2) - 2 * mu * mv - 4 * (du + dv) + 4),
    44: lambda du, mu, dv, mv: 2 + np.sqrt(2 * ((du - 1)**2 + (dv - 1)**2 + mu * mv - du * dv)),
    45: lambda du, mu, dv, mv: 2 + np.sqrt((du - dv)**2 + 2 * (du * mu + dv * mv) - 4 * (mu + mv) + 4),
    46: lambda du, mu, dv, mv: 2 + np.sqrt(2 * (du**2 + dv**2) - 16 * (du * dv) / (mu + mv) + 4),
    47: lambda du, mu, dv, mv: (2 * (du**2 + dv**2) - (mu - mv)**2) / (du + dv),
    48: lambda du, mu, dv, mv: 2 * (du**2 + dv**2) / (2 + np.sqrt(2 * (du**2 + dv**2) - 4 * (mu + mv) + 4)),
    49: lambda du, mu, dv, mv: 2 + np.sqrt(2 * (mu**2 + mv**2) + (du - dv)**2 - 4 * (du + dv) + 4),
    50: lambda du, mu, dv, mv: (du**2 + dv**2 + mu * mv - du * dv) / (du + dv),
    51: lambda du, mu, dv, mv: 2 * (mu + mv) - 4 * (mu * mv) / (du + dv),
    52: lambda du, mu, dv, mv: 2 + np.sqrt(np.sqrt(8 * (mu**4 + mv**4) - 8 * (du**2 + dv**2) + 4) - 4 * (du + dv) + 6),
    53: lambda du, mu, dv, mv: 2 + np.sqrt(np.sqrt(8 * (mu**4 + mv**4) - 8 * (du * mu + dv * mv) + 4) - 4 * (du + dv) + 6),
    54: lambda du, mu, dv, mv: 2 + np.sqrt(2 * (mu**2 + mv**2) + (du * mu + dv * mv) - (du**2 + dv**2) - 4 * (du + dv) + 4),
    55: lambda du, mu, dv, mv: 2 + np.sqrt(3 * (mu**2 + mv**2) - (du**2 + dv**2) - 4 * (mu + mv) + 4),
    56: lambda du, mu, dv, mv: ((du**2 + dv**2) * (mu + mv)) / (2 * du * dv),
    57: lambda du, mu, dv, mv: 2 + np.sqrt(2 * (mu**2 + mv**2) - 8 * (du**2 + dv**2) / (mu + mv) + 4),
    58: lambda du, mu, dv, mv: 2 + np.sqrt(2 * (mu**2 + mu * mv + mv**2) - (du * mu + dv * mv) - 4 * (du + dv) + 4),
    59: lambda du, mu, dv, mv: (2 * (mu**2 + mu * mv + mv**2) - (du**2 + dv**2)) / (mu + mv),
    60: lambda du, mu, dv, mv: 2 + np.sqrt(2 * (mu**2 + mu * mv + mv**2) - (du**2 + dv**2) - 4 * (du + dv) + 4),
    61: lambda du, mu, dv, mv: (2 * (mu**2 + mv**2)) / (2 + np.sqrt(2 * ((du - 1)**2 + (dv - 1)**2))),
    62: lambda du, mu, dv, mv: 2 + np.sqrt(mu**2 + 4 * mu * mv + mv**2 - 2 * du * dv - 4 * (du + dv) + 4),
    63: lambda du, mu, dv, mv: du + dv + mu + mv - 4 * (du * dv) / (mu + mv),
    64: lambda du, mu, dv, mv: (mu * mv * (du + dv)) / (du * dv),
    65: lambda du, mu, dv, mv: ((mu + mv) * (du * mu + dv * mv)) / (2 * mu * mv),
    66: lambda du, mu, dv, mv: (mu**2 + 4 * mu * mv + mv**2 - (du * mu + dv * mv)) / (du + dv),
    67: lambda du, mu, dv, mv: ((mu + mv) * (du * mu + dv * mv)) / (2 * du * dv),
    68: lambda du, mu, dv, mv: 2 + np.sqrt((mu - mv)**2 + 4 * du * dv - 4 * (mu + mv) + 4),
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
        np.nan_to_num(all_right_hand_sides, nan=-1000.0, copy=False)

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
        candidates_count=500,
        elite_count=75,
        survivors_count=100,
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
    # main(graph_order=16, expression_index=3)
    # print()
    # main(graph_order=16, expression_index=15)
    # print()
    # main(graph_order=16, expression_index=28)
    # print()
    # main(graph_order=16, expression_index=29)
    # print()
    # main(graph_order=16, expression_index=31)

    # for expression_index in [52, 53, 54, 55]:
    #     main(graph_order=16, expression_index=expression_index)
    
    main(graph_order=20, expression_index=42)