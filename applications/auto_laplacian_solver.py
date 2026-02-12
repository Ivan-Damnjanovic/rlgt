import pickle

import numpy as np
import torch.nn as nn
import torch.optim as optim

from rlgt.agents import (
    DeepCrossEntropyAgent,
    ExponentialRandomActionMechanism,
    PPOAgent,
    ReinforceAgent,
)
from rlgt.environments import (
    GlobalFlipEnvironment,
    LinearBuildEnvironment,
    LocalSetEnvironment,
    create_fixed_graph_generator,
)
from rlgt.graphs import CycleGraph, Graph, GraphFormat


# The expressions used to construct the right-hand sides of the inequalities from
#     V. Brankov, P. Hansen and D. Stevanović, Automated conjectures on upper bounds for the
#     largest Laplacian eigenvalue of graphs, Linear Algebra Appl. 414 (2006), 407-424.
LAPLACIAN_EXPRESSIONS = {
    1: lambda d, m: np.sqrt(4 * d**3 / m),
    2: lambda d, m: 2 * m**2 / d,
    3: lambda d, m: m**2 / d + m,
    4: lambda d, m: 2 * d**2 / m,
    5: lambda d, m: d**2 / m + m,
    6: lambda d, m: np.sqrt(m**2 + 3 * d**2),
    7: lambda d, m: d**2 / m + d,
    8: lambda d, m: np.sqrt(d * (m + 3 * d)),
    9: lambda d, m: (m + 3 * d) / 2,
    10: lambda d, m: np.sqrt(d * (d + 3 * m)),
    11: lambda d, m: 2 * m**3 / d**2,
    12: lambda d, m: np.sqrt(2 * m**2 + 2 * d**2),
    13: lambda d, m: 2 * m**4 / d**3,
    14: lambda d, m: 2 * d**3 / m**2,
    15: lambda d, m: np.sqrt(4 * m**3 / d),
    16: lambda d, m: 2 * d**4 / m**3,
    17: lambda d, m: (5 * d**4 + 11 * m**4) ** 0.25,
    18: lambda d, m: np.sqrt(2 * m**3 / d + 2 * d**2),
    19: lambda d, m: (4 * d**4 + 12 * d * m**3) ** 0.25,
    20: lambda d, m: np.sqrt(7 * d**2 + 9 * m**2) / 2,
    21: lambda d, m: np.sqrt(d**3 / m + 3 * m**2),
    22: lambda d, m: (2 * d**4 + 14 * d**2 * m**2) ** 0.25,
    23: lambda d, m: np.sqrt(d**2 + 3 * d * m),
    24: lambda d, m: (6 * d**4 + 10 * m**4) ** 0.25,
    25: lambda d, m: (3 * d**4 + 13 * d**2 * m**2) ** 0.25,
    26: lambda d, m: np.sqrt(5 * d**2 + 11 * d * m) / 2,
    27: lambda d, m: np.sqrt((3 * d**2 + 5 * d * m) / 2),
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
    38: lambda du, mu, dv, mv: 2 + np.sqrt(2 * (du - 1) ** 2 + 2 * (dv - 1) ** 2),
    39: lambda du, mu, dv, mv: 2 + np.sqrt(2 * (du**2 + dv**2) - 4 * (mu + mv) + 4),
    40: lambda du, mu, dv, mv: 2
    + np.sqrt(2 * ((mu - 1) ** 2 + (mv - 1) ** 2) + (du**2 + dv**2) - (du * mu + dv * mv)),
    41: lambda du, mu, dv, mv: 2
    + (mu + mv)
    - (du + dv)
    + np.sqrt(2 * (du**2 + dv**2) - 4 * (mu + mv) + 4),
    42: lambda du, mu, dv, mv: np.sqrt(du**2 + dv**2 + 2 * mu * mv),
    43: lambda du, mu, dv, mv: 2 + np.sqrt(3 * (mu**2 + mv**2) - 2 * mu * mv - 4 * (du + dv) + 4),
    44: lambda du, mu, dv, mv: 2
    + np.sqrt(2 * ((du - 1) ** 2 + (dv - 1) ** 2 + mu * mv - du * dv)),
    45: lambda du, mu, dv, mv: 2
    + np.sqrt((du - dv) ** 2 + 2 * (du * mu + dv * mv) - 4 * (mu + mv) + 4),
    46: lambda du, mu, dv, mv: 2 + np.sqrt(2 * (du**2 + dv**2) - 16 * (du * dv) / (mu + mv) + 4),
    47: lambda du, mu, dv, mv: (2 * (du**2 + dv**2) - (mu - mv) ** 2) / (du + dv),
    48: lambda du, mu, dv, mv: 2
    * (du**2 + dv**2)
    / (2 + np.sqrt(2 * (du**2 + dv**2) - 4 * (mu + mv) + 4)),
    49: lambda du, mu, dv, mv: 2
    + np.sqrt(2 * (mu**2 + mv**2) + (du - dv) ** 2 - 4 * (du + dv) + 4),
    50: lambda du, mu, dv, mv: 2 * (du**2 + dv**2 + mu * mv - du * dv) / (du + dv),
    51: lambda du, mu, dv, mv: 2 * (mu + mv) - 4 * (mu * mv) / (du + dv),
    52: lambda du, mu, dv, mv: 2
    + np.sqrt(np.sqrt(8 * (mu**4 + mv**4) - 8 * (du**2 + dv**2) + 4) - 4 * (du + dv) + 6),
    53: lambda du, mu, dv, mv: 2
    + np.sqrt(np.sqrt(8 * (mu**4 + mv**4) - 8 * (du * mu + dv * mv) + 4) - 4 * (du + dv) + 6),
    54: lambda du, mu, dv, mv: 2
    + np.sqrt(2 * (mu**2 + mv**2) + (du * mu + dv * mv) - (du**2 + dv**2) - 4 * (du + dv) + 4),
    55: lambda du, mu, dv, mv: 2
    + np.sqrt(3 * (mu**2 + mv**2) - (du**2 + dv**2) - 4 * (mu + mv) + 4),
    56: lambda du, mu, dv, mv: ((du**2 + dv**2) * (mu + mv)) / (2 * du * dv),
    57: lambda du, mu, dv, mv: 2
    + np.sqrt(2 * (mu**2 + mv**2) - 8 * (du**2 + dv**2) / (mu + mv) + 4),
    58: lambda du, mu, dv, mv: 2
    + np.sqrt(2 * (mu**2 + mu * mv + mv**2) - (du * mu + dv * mv) - 4 * (du + dv) + 4),
    59: lambda du, mu, dv, mv: (2 * (mu**2 + mu * mv + mv**2) - (du**2 + dv**2)) / (mu + mv),
    60: lambda du, mu, dv, mv: 2
    + np.sqrt(2 * (mu**2 + mu * mv + mv**2) - (du**2 + dv**2) - 4 * (du + dv) + 4),
    61: lambda du, mu, dv, mv: (2 * (mu**2 + mv**2))
    / (2 + np.sqrt(2 * ((du - 1) ** 2 + (dv - 1) ** 2))),
    62: lambda du, mu, dv, mv: 2
    + np.sqrt(mu**2 + 4 * mu * mv + mv**2 - 2 * du * dv - 4 * (du + dv) + 4),
    63: lambda du, mu, dv, mv: du + dv + mu + mv - 4 * (du * dv) / (mu + mv),
    64: lambda du, mu, dv, mv: (mu * mv * (du + dv)) / (du * dv),
    65: lambda du, mu, dv, mv: ((mu + mv) * (du * mu + dv * mv)) / (2 * mu * mv),
    66: lambda du, mu, dv, mv: (mu**2 + 4 * mu * mv + mv**2 - (du * mu + dv * mv)) / (du + dv),
    67: lambda du, mu, dv, mv: ((mu + mv) * (du * mu + dv * mv)) / (2 * du * dv),
    68: lambda du, mu, dv, mv: 2 + np.sqrt((mu - mv) ** 2 + 4 * du * dv - 4 * (mu + mv) + 4),
}


def compute_graph_invariant(graph_batch: Graph, expression_index: int):
    r"""
    This function computes a graph invariant of the form
    \[
        \mu - \max_{v \in V} h(d(v), m(v))
    \]
    or
    \[
        \mu - \max_{uv \in E} h(d(u), m(u), d(v), m(v))
    \]
    for a provided batch of graphs, where $\mu$ is the Laplacian spectral radius, $d(v)$ is the
    degree of a vertex $v$, $m(v)$ is the average degree of the neighbors of a vertex $v$, and $h$
    is one of the 68 right-hand side expressions from
        V. Brankov, P. Hansen and D. Stevanović, Automated conjectures on upper bounds for the
        largest Laplacian eigenvalue of graphs, Linear Algebra Appl. 414 (2006), 407-424.
    The maximum is taken over all the graph vertices $v$ or the graph edges $uv$, depending on the
    right-hand side expression. An input graph is assumed to be connected; otherwise, a score of
    -10.0 is returned.

    :param graph_batch: The provided batch of graphs, given as a `Graph` object.
    :param expression_integer: A positive `int` between 1 and 68 specifying which of the 68
        right-hand side expressions should be used.

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

    # If one of the first 32 right-hand side expressions should be used, then just compute the
    # required maximum over all the graph vertices.
    if expression_index <= 32:
        right_hand_side_batch = np.max(
            LAPLACIAN_EXPRESSIONS[expression_index](d_batch_fixed, m_batch_fixed), axis=1
        )
    # Otherwise, a maximum over all the graph edges needs to be computed, which requires additional
    # work.
    else:
        b, u, v = np.nonzero(np.triu(graph_batch.adjacency_matrix_colors, k=1))

        du = d_batch_fixed[b, u]
        mu = m_batch_fixed[b, u]
        dv = d_batch_fixed[b, v]
        mv = m_batch_fixed[b, v]

        all_right_hand_sides = LAPLACIAN_EXPRESSIONS[expression_index](du, mu, dv, mv)
        # For some right-hand side expressions, a square root argument can be negative, hence the
        # value is not defined. In this case, just ignore this value while taking the maximum.
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


def solve_dce(graph_order: int, expression_index: int):
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
            graph_invariant=lambda graph_batch: compute_graph_invariant(
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

    print(f"Conjecture {expression_index}:")
    agent.reset()

    while True:
        agent.step()
        print(f"Learning iterations: {agent.step_count}. Best score: {agent.best_score:.3f}.")

        if agent.best_score > 0.0001:
            print("Success! The following graph is a solution:")
            solution = agent.best_graph.adjacency_matrix_colors
            print(solution)

            try:
                with open("applications/auto_laplacian_solutions.pkl", "rb") as opened_file:
                    all_solutions = pickle.load(opened_file)
            except:
                all_solutions = []

            all_solutions.append(solution)

            with open("applications/auto_laplacian_solutions.pkl", "wb") as opened_file:
                pickle.dump(all_solutions, opened_file)

            break

        if agent.step_count >= 2000:
            agent.reset()


def solve_reinforce(graph_order: int, expression_index: int):
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
            graph_invariant=lambda graph_batch: compute_graph_invariant(
                graph_batch=graph_batch, expression_index=expression_index
            ),
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
        random_action_mechanism=ExponentialRandomActionMechanism(
            initial_random_action_probability=0.005,
            waiting_period=10,
            multiplicative_factor=1.1,
            maximum_random_action_probability=0.100,
        ),
    )

    print(f"Conjecture {expression_index}:")
    agent.reset()

    while True:
        agent.step()
        print(f"Learning iterations: {agent.step_count}. Best score: {agent.best_score:.3f}.")

        if agent.best_score > 0.0001:
            print("Success! The following graph is a solution:")
            solution = agent.best_graph.adjacency_matrix_colors
            print(solution)

            try:
                with open("applications/auto_laplacian_solutions.pkl", "rb") as opened_file:
                    all_solutions = pickle.load(opened_file)
            except:
                all_solutions = []

            all_solutions.append(solution)

            with open("applications/auto_laplacian_solutions.pkl", "wb") as opened_file:
                pickle.dump(all_solutions, opened_file)

            break

        if agent.step_count >= 2000:
            agent.reset()


def solve_ppo(graph_order: int, expression_index: int):
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
            graph_invariant=lambda graph_batch: compute_graph_invariant(
                graph_batch=graph_batch, expression_index=expression_index
            ),
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
            list(policy_network.parameters()) + list(value_network.parameters()), lr=0.0005
        ),
        random_action_mechanism=ExponentialRandomActionMechanism(
            initial_random_action_probability=0.005,
            waiting_period=10,
            multiplicative_factor=1.1,
            maximum_random_action_probability=0.100,
        ),
    )

    print(f"Conjecture {expression_index}:")
    agent.reset()

    while True:
        agent.step()
        print(f"Learning iterations: {agent.step_count}. Best score: {agent.best_score:.3f}.")

        if agent.best_score > 0.0001:
            print("Success! The following graph is a solution:")
            solution = agent.best_graph.adjacency_matrix_colors
            print(solution)

            try:
                with open("applications/auto_laplacian_solutions.pkl", "rb") as opened_file:
                    all_solutions = pickle.load(opened_file)
            except:
                all_solutions = []

            all_solutions.append(solution)

            with open("applications/auto_laplacian_solutions.pkl", "wb") as opened_file:
                pickle.dump(all_solutions, opened_file)

            break

        if agent.step_count >= 2000:
            agent.reset()


if __name__ == "__main__":
    try:
        with open("applications/auto_laplacian_solutions.pkl", "rb") as opened_file:
            all_solutions = pickle.load(opened_file)
    except:
        all_solutions = []

    converted_solutions = []
    for solution in all_solutions:
        g = Graph.from_adjacency_matrix(solution)
        bitmask = g.bitmask_out[0]
        converted_solutions.append(bitmask)

    with open("applications/auto_laplacian_solutions.txt", "w") as opened_file:
        for solution in converted_solutions:
            line = " ".join(str(x) for x in solution)
            opened_file.write(line + "\n")
