import numpy as np
import torch.nn as nn
import torch.optim as optim
from sage.all import *

# Make sure not to have a collision with the ``SageMath`` package names!
from rlgt import agents as rlgt_agents
from rlgt import environments as rlgt_environments
from rlgt import graphs as rlgt_graphs


def mostar_index(graph_batch: rlgt_graphs.Graph) -> np.ndarray:
    r"""
    This function computes the Mostar index for a provided batch of graphs, i.e., the graph
    invariant
    \[
        \sum_{uv \in E} | n(u, v) - n(v, u) |,
    \]
    where $n(u, v)$ is the number of vertices closer to $u$ than to $v$ and $n(v, u)$ is defined
    analogously. An input graph is assumed to be connected; otherwise, a score of -2000.0 is
    returned.

    :param graph_batch: The provided batch of graphs, given as a `Graph` object.

    :return: The computed batch of Mostar index values, given as a `numpy.ndarray` of type
        `numpy.float32`.
    """

    # Initialize a `numpy.ndarray` of type `numpy.float32` where all the computed Mostar index
    # values should be stored.
    scores = np.empty(graph_batch.batch_size, dtype=np.float32)

    # For each of the graphs in the batch...
    for index in range(graph_batch.batch_size):
        g = Graph(matrix(graph_batch.adjacency_matrix_colors[index]))

        # If the graph is not connected, punish it.
        if not g.is_connected():
            scores[index] = -2000.0
            continue

        # Compute the vertex transmissions.
        transmissions = [sum(row) for row in g.distance_matrix().rows()]

        # Compute the Mostar index.
        mostar = 0
        for u, v, _ in g.edges():
            mostar += abs(transmissions[u] - transmissions[v])

        scores[index] = mostar

    return scores


def attempt(graph_order: int, stop_when_conjectured_maximum: bool = True):
    r"""
    This function attempts to maximize the Mostar index on the set of connected graphs of a
    specified order. The Mostar index is defined as
    \[
        \sum_{uv \in E} | n(u, v) - n(v, u) |,
    \]
    where $n(u, v)$ is the number of vertices closer to $u$ than to $v$ and $n(v, u)$ is defined
    analogously, as introduced in

    * T. Došlić, I. Martinjak, R. Škrekovski, S. Tipurić Spužević and I. Zubac, Mostar index, J.
      Math. Chem. 56 (2018), 2995-3013.

    The optimization process can be stopped either when the conjectured maximum value is reached or
    when it is surpassed, depending on the configuration.

    :param graph_order: A positive `int` not less than 2 specifying the graph order.
    :param stop_when_conjectured_maximum: A `bool` that determines whether the optimization process
        should stop when the conjectured maximum value is reached (`True`) or when it is surpassed
        (`False`). The default value is `True`.
    """

    # Compute the conjectured maximum value.
    temp = graph_order // 3
    conjectured_maximum = temp * (graph_order - temp) * (graph_order - temp - 1)

    # Compute the optimization threshold value depending on the optimization goal configuration.
    if stop_when_conjectured_maximum:
        threshold_value = conjectured_maximum - 0.5
    else:
        threshold_value = conjectured_maximum + 0.5

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
            graph_invariant=mostar_index,
            graph_order=graph_order,
        ),
        policy_network=policy_network,
        optimizer=optim.Adam(policy_network.parameters(), lr=0.003),
        random_action_mechanism=rlgt_agents.ExponentialRandomActionMechanism(
            initial_random_action_probability=0.005,
            waiting_period=10,
            multiplicative_factor=1.1,
            maximum_random_action_probability=0.200,
        ),
    )

    print(f"Conjectured maximum value: {conjectured_maximum}.")
    print("Starting...")
    agent.reset()

    while True:
        agent.step()
        print(f"Learning iterations: {agent.step_count}. Best score: {agent.best_score:.3f}.")

        if agent.best_score > threshold_value:
            print(f"Success! The following graph has a Mostar index of {agent.best_score}:")
            print(agent.best_graph.adjacency_matrix_colors)

            break

        if agent.step_count >= 500:
            print("Restarting...")
            agent.reset()


if __name__ == "__main__":
    attempt(graph_order=21)
