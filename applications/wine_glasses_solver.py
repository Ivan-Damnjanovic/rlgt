import numpy as np
import torch.nn as nn
import torch.optim as optim
from sage.all import *

# Make sure not to have a collision with the ``SageMath`` package names!
from rlgt import agents as rlgt_agents
from rlgt import environments as rlgt_environments
from rlgt import graphs as rlgt_graphs


def graph_invariant(graph_batch: rlgt_graphs.Graph) -> np.ndarray:
    r"""
    This function computes the graph invariant
    \[
        \mathcal{E} - 2 \nu \sqrt{\Delta}
    \]
    for a provided batch of graphs, where $\mathcal{E}$ is the graph energy, $\nu$ is the matching
    number, and $\Delta$ is the maximum vertex degree, in accordance with the conjectured
    inequality from

    * S. Akbari, A. Alazemi and M. Anđelić, Upper bounds on the energy of graphs in terms of
      matching number, Appl. Anal. Discrete Math. 15 (2021), 444-459.

    An input graph is assumed to be connected and its maximum vertex degree must be below 6;
    otherwise, a score of -2000.0 is returned.

    :param graph_batch: The provided batch of graphs, given as a `Graph` object.

    :return: The computed batch of graph invariant values, given as a `numpy.ndarray` of type
        `numpy.float32`.
    """

    # Initialize a `numpy.ndarray` of type `numpy.float32` where all the computed graph invariant
    # values should be stored.
    scores = np.empty(graph_batch.batch_size, dtype=np.float32)

    # For each of the graphs in the batch...
    for index in range(graph_batch.batch_size):
        g = Graph(matrix(graph_batch.adjacency_matrix_colors[index]))

        # If the graph is not connected, punish it.
        if not g.is_connected():
            scores[index] = -2000.0
            continue

        # Compute the maximum vertex degree.
        delta = max(g.degree())

        # If the maximum degree is not below 6, punish the graph.
        if delta > 5:
            scores[index] = -2000.0
            continue

        # Compute the matching number.
        nu = len(g.matching())

        # Compute the graph energy.
        eigenvalues = g.adjacency_matrix().eigenvalues()
        energy = sum(abs(eigenvalue) for eigenvalue in eigenvalues)

        # Finally, compute the graph invariant value.
        scores[index] = energy - 2 * nu * sqrt(delta)

    return scores


def solve(graph_order: int):
    r"""
    This function attempts to find a counterexample of a configured order to the conjectured
    inequality
    \[
        \mathcal{E} \le 2 \nu \sqrt{\Delta}
    \]
    where $\mathcal{E}$ is the graph energy, $\nu$ is the matching number, and $\Delta$ is the
    maximum vertex degree, from

    * S. Akbari, A. Alazemi and M. Anđelić, Upper bounds on the energy of graphs in terms of
      matching number, Appl. Anal. Discrete Math. 15 (2021), 444-459.

    The simple graphs in the conjecture are required to be connected and of maximum degree below 6,
    and they should not be a cycle graph of order 3, 5 or 7.

    :param graph_order: A positive `int` not less than 2 specifying the order of the graphs that
        should be built for the purpose of potentially finding a counterexample.
    """

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

    print("Starting...")
    agent.reset()

    while True:
        agent.step()
        print(f"Learning iterations: {agent.step_count}. Best score: {agent.best_score:.3f}.")

        if agent.best_score > 0.0001:
            best_graph = agent.best_graph

            print("Success! The following graph is a solution:")
            print(best_graph.adjacency_matrix_colors)

            # Save the solution in the bitmask format as a new line.
            with open("applications/wine_glasses_solutions.txt", "a") as opened_file:
                line = " ".join(str(entry) for entry in best_graph.bitmask_out[-1])
                opened_file.write(line + "\n")

            break

        if agent.step_count >= 100 and agent.best_score < -100.0:
            print("Restarting...")
            agent.reset()

        if agent.step_count >= 500:
            print("Restarting...")
            agent.reset()


if __name__ == "__main__":
    solve(graph_order=14)
