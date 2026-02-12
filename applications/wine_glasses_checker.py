import numpy as np
from sage.all import *

# Make sure not to have a collision with the ``SageMath`` package names!
import rlgt.graphs as rlgt_graphs


def check(adjacency_matrix):
    r"""
    This function determines whether a connected simple graph with maximum vertex degree below 6
    represented through an adjacency matrix is a counterexample to the conjectured inequality
    \[
        \mathcal{E} \le 2 \mu \sqrt{\Delta}
    \]
    where $\mathcal{E}$ is the graph energy, $\mu$ is the matching number, and $\Delta$ is the
    maximum vertex degree, from

    * S. Akbari, A. Alazemi and M. Anđelić, Upper bounds on the energy of graphs in terms of
      matching number, Appl. Anal. Discrete Math. 15 (2021), 444-459.

    If the provided graph is indeed a counterexample, then its adjacency matrix should be printed.

    :param adjacency_matrix: The adjacency matrix of the provided graph, given as a `numpy.ndarray`
        matrix of type `numpy.uint8`.
    """

    g = Graph(matrix(ZZ, adjacency_matrix))

    # The graph must be on at least 8 vertices and it must be connected.
    assert g.order() >= 8 and g.is_connected()

    # Compute the maximum vertex degree and make sure that it is below 6.
    delta = max(g.degree())
    assert delta < 6

    # Compute the matching number.
    mu = len(g.matching())

    # Compute the graph energy.
    eigenvalues = g.adjacency_matrix().eigenvalues()
    energy = sum(abs(eval) for eval in eigenvalues)

    if float(energy - 2 * mu * sqrt(delta)) > 0.005:
        print(adjacency_matrix)


if __name__ == "__main__":
    # Iterate through all the discovered solution graphs.
    with open("applications/wine_glasses_solutions.txt", "r") as opened_file:
        for line in opened_file:
            bitmask = np.array([[int(entry) for entry in line.split()]], dtype=np.uint64)
            adjacency_matrix = rlgt_graphs.Graph.from_bitmask(bitmask).adjacency_matrix_colors

            check(adjacency_matrix)
