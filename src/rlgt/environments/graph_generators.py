"""
This `Python` module defines several factory functions for constructing graph generator functions.
These generators implement various mechanisms for producing batches of fully colored
$k$-edge-colored looped complete graphs of a specified batch size.
"""

from typing import Callable, Optional, Union

import numpy as np

from ..graphs.graph import Graph
from ..graphs.graph_formats import FlattenedOrdering, GraphFormat
from ..graphs.utils import graph_order_to_flattened_length


GraphGenerator = Callable[[int], Graph]
"""
This is a type alias for representing graph generator functions that accept a single integer
argument specifying the batch size and return a `Graph` object representing a batch of fully
colored $k$-edge-colored looped complete graphs of that size. The graphs in the batch are generated
using an internal mechanism defined by the generator function, and the generation process need not
be deterministic.
"""


def create_fixed_graph_generator(
    fixed_graph: Graph, graph_format: GraphFormat = GraphFormat.FLATTENED_ROW_MAJOR_COLORS
) -> GraphGenerator:
    """
    This function creates a `GraphGenerator` function that outputs batches of fully colored
    $k$-edge-colored looped complete graphs, where all the graphs are identical to a user-provided
    ``fixed_graph``. The user can also configure the starting graph format of the output batch.

    :param fixed_graph: A fully colored `Graph` object that all graphs in the output batch should
        be identical to.
    :param graph_format: An item of the `GraphFormat` enumeration specifying the format in which
        the output batch of graphs should be initialized. The default value is
        `GraphFormat.FLATTENED_ROW_MAJOR_COLORS`, i.e., the flattened row-major format with color
        numbers.

    :return: The created `GraphGenerator` function.
    """

    # Depending on the chosen graph format, select the format representation to be used by the
    # graph generator, as well as the corresponding constructor for creating the output batch of
    # graphs.
    if graph_format == GraphFormat.FLATTENED_ROW_MAJOR_COLORS:
        input_representation = fixed_graph.flattened_row_major_colors
        constructor = lambda representation: Graph(
            edge_colors=fixed_graph.edge_colors,
            is_directed=fixed_graph.is_directed,
            allow_loops=fixed_graph.allow_loops,
            flattened_row_major_colors=representation,
        )

    elif graph_format == GraphFormat.FLATTENED_CLOCKWISE_COLORS:
        input_representation = fixed_graph.flattened_clockwise_colors
        constructor = lambda representation: Graph(
            edge_colors=fixed_graph.edge_colors,
            is_directed=fixed_graph.is_directed,
            allow_loops=fixed_graph.allow_loops,
            flattened_clockwise_colors=representation,
        )

    elif graph_format == GraphFormat.FLATTENED_ROW_MAJOR_BINARY:
        input_representation = fixed_graph.flattened_row_major_binary
        constructor = lambda representation: Graph(
            edge_colors=fixed_graph.edge_colors,
            is_directed=fixed_graph.is_directed,
            allow_loops=fixed_graph.allow_loops,
            flattened_row_major_binary=representation,
        )

    elif graph_format == GraphFormat.FLATTENED_CLOCKWISE_BINARY:
        input_representation = fixed_graph.flattened_clockwise_binary
        constructor = lambda representation: Graph(
            edge_colors=fixed_graph.edge_colors,
            is_directed=fixed_graph.is_directed,
            allow_loops=fixed_graph.allow_loops,
            flattened_clockwise_binary=representation,
        )

    elif graph_format == GraphFormat.BITMASK_OUT:
        input_representation = fixed_graph.bitmask_out
        constructor = lambda representation: Graph(
            edge_colors=fixed_graph.edge_colors,
            is_directed=fixed_graph.is_directed,
            allow_loops=fixed_graph.allow_loops,
            bitmask_out=representation,
        )

    elif graph_format == GraphFormat.BITMASK_IN:
        input_representation = fixed_graph.bitmask_in
        constructor = lambda representation: Graph(
            edge_colors=fixed_graph.edge_colors,
            is_directed=fixed_graph.is_directed,
            allow_loops=fixed_graph.allow_loops,
            bitmask_in=representation,
        )

    elif graph_format == GraphFormat.ADJACENCY_MATRIX_COLORS:
        input_representation = fixed_graph.adjacency_matrix_colors
        constructor = lambda representation: Graph(
            edge_colors=fixed_graph.edge_colors,
            is_directed=fixed_graph.is_directed,
            allow_loops=fixed_graph.allow_loops,
            adjacency_matrix_colors=representation,
        )

    else:
        input_representation = fixed_graph.adjacency_matrix_binary
        constructor = lambda representation: Graph(
            edge_colors=fixed_graph.edge_colors,
            is_directed=fixed_graph.is_directed,
            allow_loops=fixed_graph.allow_loops,
            adjacency_matrix_binary=representation,
        )

    # Create the graph generator.
    def result(batch_size: int) -> Graph:
        format_representation = np.empty(
            (batch_size, *input_representation.shape), dtype=input_representation.dtype
        )
        format_representation[:] = input_representation

        return constructor(format_representation)

    return result


def create_choose_two_graph_generator(
    first_graph: Graph,
    second_graph: Graph,
    second_graph_probability: float,
    graph_format: GraphFormat = GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
    random_generator: Optional[np.random.Generator] = None,
) -> GraphGenerator:
    """
    This function creates a `GraphGenerator` function that outputs batches of fully colored
    $k$-edge-colored looped complete graphs, where each graph in the batch is assigned to be equal
    to either one of two user-provided graphs according to a given probability. The two graphs must
    have the same number of edge colors, the same order, and the same type (both directed or both
    undirected, and both either allow loops or not). The user can also configure the starting graph
    format for the output batch.

    :param first_graph: A fully colored `Graph` object representing the first graph that the output
        batch graphs could be equal to.
    :param second_graph: A fully colored `Graph` object representing the second graph that the
        output batch graphs could be equal to.
    :param second_graph_probability: A `float` in the interval $[0, 1]$ specifying the probability
        that a graph in the output batch is equal to `second_graph`.
    :param graph_format: An item of the `GraphFormat` enumeration specifying the format in which
        the output batch of graphs should be initialized. The default value is
        `GraphFormat.FLATTENED_ROW_MAJOR_COLORS`, i.e., the flattened row-major format with color
        numbers.
    :param random_generator: Either `None`, or a `numpy.random.Generator` used for probabilistic
        decisions. If `None`, a default generator will be used. The default value is `None`.

    :return: The created `GraphGenerator` function.
    """

    # If the ``random_generator`` argument is `None`, then use a default `np.random.Generator`.
    if random_generator is None:
        random_generator = np.random.default_rng()  # pragma: no cover

    # Depending on the chosen graph format, select the two format representations to be used by the
    # graph generator, as well as the corresponding constructor for creating the output batch of
    # graphs.
    if graph_format == GraphFormat.FLATTENED_ROW_MAJOR_COLORS:
        input_representation_1 = first_graph.flattened_row_major_colors
        input_representation_2 = second_graph.flattened_row_major_colors
        constructor = lambda representation: Graph(
            edge_colors=first_graph.edge_colors,
            is_directed=first_graph.is_directed,
            allow_loops=first_graph.allow_loops,
            flattened_row_major_colors=representation,
        )

    elif graph_format == GraphFormat.FLATTENED_CLOCKWISE_COLORS:
        input_representation_1 = first_graph.flattened_clockwise_colors
        input_representation_2 = second_graph.flattened_clockwise_colors
        constructor = lambda representation: Graph(
            edge_colors=first_graph.edge_colors,
            is_directed=first_graph.is_directed,
            allow_loops=first_graph.allow_loops,
            flattened_clockwise_colors=representation,
        )

    elif graph_format == GraphFormat.FLATTENED_ROW_MAJOR_BINARY:
        input_representation_1 = first_graph.flattened_row_major_binary
        input_representation_2 = second_graph.flattened_row_major_binary
        constructor = lambda representation: Graph(
            edge_colors=first_graph.edge_colors,
            is_directed=first_graph.is_directed,
            allow_loops=first_graph.allow_loops,
            flattened_row_major_binary=representation,
        )

    elif graph_format == GraphFormat.FLATTENED_CLOCKWISE_BINARY:
        input_representation_1 = first_graph.flattened_clockwise_binary
        input_representation_2 = second_graph.flattened_clockwise_binary
        constructor = lambda representation: Graph(
            edge_colors=first_graph.edge_colors,
            is_directed=first_graph.is_directed,
            allow_loops=first_graph.allow_loops,
            flattened_clockwise_binary=representation,
        )

    elif graph_format == GraphFormat.BITMASK_OUT:
        input_representation_1 = first_graph.bitmask_out
        input_representation_2 = second_graph.bitmask_out
        constructor = lambda representation: Graph(
            edge_colors=first_graph.edge_colors,
            is_directed=first_graph.is_directed,
            allow_loops=first_graph.allow_loops,
            bitmask_out=representation,
        )

    elif graph_format == GraphFormat.BITMASK_IN:
        input_representation_1 = first_graph.bitmask_in
        input_representation_2 = second_graph.bitmask_in
        constructor = lambda representation: Graph(
            edge_colors=first_graph.edge_colors,
            is_directed=first_graph.is_directed,
            allow_loops=first_graph.allow_loops,
            bitmask_in=representation,
        )

    elif graph_format == GraphFormat.ADJACENCY_MATRIX_COLORS:
        input_representation_1 = first_graph.adjacency_matrix_colors
        input_representation_2 = second_graph.adjacency_matrix_colors
        constructor = lambda representation: Graph(
            edge_colors=first_graph.edge_colors,
            is_directed=first_graph.is_directed,
            allow_loops=first_graph.allow_loops,
            adjacency_matrix_colors=representation,
        )

    else:
        input_representation_1 = first_graph.adjacency_matrix_binary
        input_representation_2 = second_graph.adjacency_matrix_binary
        constructor = lambda representation: Graph(
            edge_colors=first_graph.edge_colors,
            is_directed=first_graph.is_directed,
            allow_loops=first_graph.allow_loops,
            adjacency_matrix_binary=representation,
        )

    # Create the graph generator.
    def result(batch_size: int) -> Graph:
        format_representation = np.empty(
            (batch_size, *input_representation_1.shape), dtype=input_representation_1.dtype
        )
        format_representation[:] = input_representation_1
        # Assign all the graphs in the batch to be equal to the second provided graph with a
        # probability of ``second_graph_probability``.
        format_representation[
            random_generator.random(size=(batch_size,)) < second_graph_probability
        ] = input_representation_2

        return constructor(format_representation)

    return result


def create_edge_perturbation_graph_generator(
    initial_graph: Graph,
    edge_perturbation_probability: float,
    color_selection_probabilities: Union[np.ndarray, float, None] = None,
    flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
    random_generator: Optional[np.random.Generator] = None,
) -> GraphGenerator:
    """
    This function creates a `GraphGenerator` function that outputs batches of fully colored
    $k$-edge-colored looped complete graphs where all graphs are initially equal to a provided
    graph, and then each edge (resp. arc) may be perturbed with a given probability. The edge
    (resp. arc) perturbation consists of recoloring it with a randomly chosen color between 0 and
    $k - 1$ according to a discrete probability distribution. Here, $k$ is determined from the
    provided initial graph. The output batch is initialized in one of the two flattened formats
    with color numbers, which can be selected via the `flattened_ordering` argument.

    :param initial_graph: A fully colored `Graph` object representing the base graph that all
        graphs in the output batch are initialized to before edge (resp. arc) perturbations.
    :param edge_perturbation_probability: A `float` from the interval $[0, 1]$ specifying the
        probability that any given edge (resp. arc) is subjected to a perturbation.
    :param color_selection_probabilities: This argument defines how the new color for a perturbed
        edge (resp. arc) is chosen. Three options are allowed:

        1. A `numpy.ndarray` of length equal to the number of proper edge colors in
           `initial_graph`. The entries must be nonnegative and sum to 1. Each entry represents the
           probability of selecting the corresponding color (in ascending order).
        2. A single `float` (only if `initial_graph` has exactly two proper edge colors)
           representing the probability of selecting color 1; color 0 is chosen with complementary
           probability.
        3. `None` (default), which selects the new color uniformly at random from all possible
           colors.

    :param flattened_ordering: An item of the `FlattenedOrdering` enumeration specifying whether
        the output batch should be initialized in the flattened row-major or flattened clockwise
        format with color numbers. The default value is `FlattenedOrdering.ROW_MAJOR`.
    :param random_generator: Either `None`, or a `numpy.random.Generator` used for probabilistic
        decisions. If `None`, a default generator will be used. The default value is `None`.

    :return: The created `GraphGenerator` function.
    """

    # If the ``random_generator`` argument is `None`, then use a default `np.random.Generator`.
    if random_generator is None:
        random_generator = np.random.default_rng()  # pragma: no cover

    if flattened_ordering == FlattenedOrdering.ROW_MAJOR:
        input_representation = initial_graph.flattened_row_major_colors
    else:
        input_representation = initial_graph.flattened_clockwise_colors

    # Create the graph generator.
    def result(batch_size: int) -> Graph:
        flattened_colors = np.empty((batch_size, input_representation.shape[0]), dtype=np.uint8)
        flattened_colors[:] = input_representation

        # Perform an edge (resp. arc) perturbation with a probability of
        # ``edge_perturbation_probability``.
        mask = random_generator.random(size=flattened_colors.shape) < edge_perturbation_probability
        entry_count = np.count_nonzero(mask)

        # If the argument ``color_selection_probabilities`` is `None`, just use the uniform
        # probability distribution.
        if color_selection_probabilities is None:
            flattened_colors[mask] = random_generator.integers(
                low=0, high=initial_graph.edge_colors, size=entry_count, dtype=np.uint8
            )
        # If the argument ``color_selection_probabilities`` is a `float`, then use this `float` as
        # the probability for coloring an edge (resp. arc) with the color 1.
        elif isinstance(color_selection_probabilities, float):
            flattened_colors[mask] = (
                random_generator.random(size=entry_count) < color_selection_probabilities
            ).astype(np.uint8)
        # If the argument ``color_selection_probabilities`` is a `np.ndarray`, then use the
        # `np.random.Generator.choice` method.
        else:
            flattened_colors[mask] = random_generator.choice(
                np.arange(initial_graph.edge_colors, dtype=np.uint8),
                size=entry_count,
                p=color_selection_probabilities,
            )

        return Graph.from_flattened(
            flattened=flattened_colors,
            flattened_ordering=flattened_ordering,
            edge_colors=initial_graph.edge_colors,
            is_directed=initial_graph.is_directed,
            allow_loops=initial_graph.allow_loops,
        )

    return result


def create_random_graph_generator(
    graph_order: int,
    color_selection_probabilities: Union[np.ndarray, float, None] = None,
    flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
    edge_colors: int = 2,
    is_directed: bool = False,
    allow_loops: bool = False,
    random_generator: Optional[np.random.Generator] = None,
) -> GraphGenerator:
    """
    This function creates a `GraphGenerator` function that outputs batches of fully colored
    $k$-edge-colored looped complete graphs generated randomly. Each edge (resp. arc) in each graph
    is assigned a color independently according to a specified discrete probability distribution.
    The user can configure the graph order, number of proper edge colors, directionality, and loop
    allowance. The output batch is initialized in one of the two flattened formats with color
    numbers, which can be selected via the `flattened_ordering` argument.

    :param graph_order: A positive `int` specifying the order of the graphs to be generated.
    :param color_selection_probabilities: This argument specifies the probability distribution for
        assigning a color to each edge (resp. arc). Three options are allowed:

        1. A `numpy.ndarray` of length equal to the number of proper edge colors in
           `initial_graph`. The entries must be nonnegative and sum to 1. Each entry represents the
           probability of selecting the corresponding color (in ascending order).
        2. A single `float` (only if `initial_graph` has exactly two proper edge colors)
           representing the probability of selecting color 1; color 0 is chosen with complementary
           probability.
        3. `None` (default), which selects the new color uniformly at random from all possible
           colors.

    :param flattened_ordering: An item of the `FlattenedOrdering` enumeration specifying whether
        the output batch should be initialized in the flattened row-major or flattened clockwise
        format with color numbers. The default value is `FlattenedOrdering.ROW_MAJOR`.
    :param edge_colors: A positive `int` (not below 2) specifying the number of proper edge colors
        in the graphs to be generated. The default value is 2.
    :param is_directed: A `bool` indicating whether the graphs to be generated are directed. The
        default value is `False`.
    :param allow_loops: A `bool` indicating whether loops are allowed in the graphs to be
        generated. The default value is `False`.
    :param random_generator: Either `None`, or a `numpy.random.Generator` used for probabilistic
        decisions. If `None`, a default generator will be used. The default value is `None`.

    :return: The created `GraphGenerator` function.
    """

    # If the ``random_generator`` argument is `None`, then use a default `np.random.Generator`.
    if random_generator is None:
        random_generator = np.random.default_rng()  # pragma: no cover

    flattened_length = graph_order_to_flattened_length(
        graph_order=graph_order,
        is_directed=is_directed,
        allow_loops=allow_loops,
    )

    # Create the graph generator.
    def result(batch_size: int) -> Graph:
        # If the argument ``color_selection_probabilities`` is `None`, just use the uniform
        # probability distribution.
        if color_selection_probabilities is None:
            flattened_colors = random_generator.integers(
                low=0, high=edge_colors, size=(batch_size, flattened_length), dtype=np.uint8
            )
        # If the argument ``color_selection_probabilities`` is a `float`, then use this `float` as
        # the probability for coloring an edge (resp. arc) with the color 1.
        elif isinstance(color_selection_probabilities, float):
            flattened_colors = (
                random_generator.random(size=(batch_size, flattened_length))
                < color_selection_probabilities
            ).astype(np.uint8)
        # If the argument ``color_selection_probabilities`` is a `np.ndarray`, then use the
        # `np.random.Generator.choice` method.
        else:
            flattened_colors = random_generator.choice(
                np.arange(edge_colors, dtype=np.uint8),
                size=(batch_size, flattened_length),
                p=color_selection_probabilities,
            )

        return Graph.from_flattened(
            flattened=flattened_colors,
            flattened_ordering=flattened_ordering,
            edge_colors=edge_colors,
            is_directed=is_directed,
            allow_loops=allow_loops,
        )

    return result
