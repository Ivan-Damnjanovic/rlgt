"""
This ``Python`` module contains several functions that create graph generator functions, which rely
on various mechanisms to construct batches of fully colored $k$-edge-colored looped complete graphs
of a given batch size.
"""

from typing import Callable, Optional, Union

import numpy as np

from ..graphs.graph import Graph
from ..graphs.graph_formats import FlattenedOrdering, GraphFormat
from ..graphs.utils import graph_order_to_flattened_length


GraphGenerator = Callable[[int], Graph]
"""
This is the type alias for the graph generator functions, which accept one integer argument that
represents the batch size, and output a `Graph` object that represents the generated batch of fully
colored $k$-edge-colored looped complete graphs of the given batch size. The graphs in the batch
are generated through some internal mechanism defined by the generator function. The generation
process need not be deterministic.
"""


def create_fixed_graph_generator(
    fixed_graph: Graph, graph_format: GraphFormat = GraphFormat.FLATTENED_ROW_MAJOR_COLORS
) -> GraphGenerator:
    """
    This function creates a `GraphGenerator` function that outputs batches of fully colored
    $k$-edge-colored looped complete graphs where all the graphs are equal to some provided graph.
    This provided graph is selectable and the user can also configure the starting graph format
    that the output batch of graphs should get initialized in.

    :param fixed_graph: A `Graph` object that represents the provided graph that all the graphs in
        the output batch should be equal to. This graph must be fully colored.
    :param graph_format: The starting graph format that the output batch of graphs should get
        initialized in, given as an item of the `GraphFormat` enumeration. The default value is
        `GraphFormat.FLATTENED_ROW_MAJOR_COLORS`, i.e., the output batch should get initialized in
        the flattened row-major format with color numbers by default.

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
    rng: Optional[np.random.Generator] = None,
) -> GraphGenerator:
    """
    This function creates a `GraphGenerator` function that outputs batches of fully colored
    $k$-edge-colored looped complete graphs where all the graphs are equal to one of the two
    provided graphs. These two provided graphs are selectable and each of the graphs in the output
    batch is assigned to be equal to one of these two graphs with respect to a given probability.
    The two provided graphs must have the same number of proper edges colors, the same order and
    the same type: they should both be directed or both be undirected, and they should both either
    all allow loops or not allow loops. The user can also configure the starting graph format that
    the output batch of graphs should get initialized in.

    :param first_graph: A `Graph` object that represents the first provided graph that the graphs
        in the output batch could be equal to. This graph must be fully colored.
    :param second_graph: A `Graph` object that represents the second provided graph that the graphs
        in the output batch could be equal to. This graph must be fully colored.
    :param second_graph_probability: A `float` from the interval $[0, 1]$ that describes the
        probability of each graph from the output batch to be assigned to be equal to the second
        provided graph.
    :param graph_format: The starting graph format that the output batch of graphs should get
        initialized in, given as an item of the `GraphFormat` enumeration. The default value is
        `GraphFormat.FLATTENED_ROW_MAJOR_COLORS`, i.e., the output batch should get initialized in
        the flattened row-major format with color numbers by default.
    :param rng: Either `None`, or the `numpy.random.Generator` object that represents the random
        number generator used for all the probabilistic decisions. If this argument is `None`, then
        a default `numpy.random.Generator` object will be used. The default value is `None`.

    :return: The created `GraphGenerator` function.
    """

    # If the ``rng`` argument is `None`, then use a default `np.random.Generator`.
    if rng is None:
        rng = np.random.default_rng()  # pragma: no cover

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
        format_representation[rng.random(size=(batch_size,)) < second_graph_probability] = (
            input_representation_2
        )

        return constructor(format_representation)

    return result


def create_edge_perturbation_graph_generator(
    initial_graph: Graph,
    edge_perturbation_probability: float,
    color_selection_probabilities: Union[np.ndarray, float, None] = None,
    flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
    rng: Optional[np.random.Generator] = None,
) -> GraphGenerator:
    """
    This function creates a `GraphGenerator` function that outputs batches of fully colored
    $k$-edge-colored looped complete graphs where all the graphs are initially equal to some
    provided graph, and then an edge (resp. arc) perturbation is performed with a given
    probability. The edge (resp. arc) perturbation consists of recoloring the edge (resp. arc) with
    a randomly chosen color between 0 and $k - 1$ with respect to a given discrete probability
    distribution. Here, the value $k$ is extracted from the provided initial graph. The output
    batch of graphs is initialized in one of the two flattened formats with color numbers, and the
    user can configure which of these two formats should get used.

    :param initial_graph: A `Graph` object that represents the provided graph that all the graphs
        in the output batch should get initialized to, before the edge (resp. arc) perturbation is
        performed. This graph must be fully colored.
    :param edge_perturbation_probability: A `float` from the interval $[0, 1]$ that describes the
        probability of each edge (resp. arc) from each of the graphs in the output batch to be
        subjected to an edge (resp. arc) perturbation.
    :param color_selection_probabilities: Either a `numpy.ndarray` list of type `numpy.floating`, a
        `float` value, or `None`. This argument is used to describe the discrete probability
        distribution for coloring the edges (resp. arcs) that are subjected to a perturbation. If
        the argument is a `numpy.ndarray` list, then the length of this array must be the same as
        the number of proper edge colors in the ``initial_graph`` object, and the list entries must
        be nonnegative numbers that sum up to 1. These entries are then used as the probabilities
        for each of the edge colors to be selected, in the ascending order. Alternatively, if the
        ``initial_graph`` object has two proper edge colors, then this argument can also be a
        `float` that represents the probability for the color 1 to be selected. Finally, the
        argument can also be `None`, in which case the uniform probability distribution is used,
        regardless of the number of proper edge colors in the ``initial_graph`` object. The default
        value is `None`.
    :param flattened_ordering: An item of the `FlattenedOrdering` enumeration that describes
        whether the flattened row-major format with color numbers or the flattened clockwise format
        with color numbers should be used to initialize the output batch of graphs. The default
        value is `FlattenedOrdering.ROW_MAJOR`, i.e., the output batch should get initialized in
        the flattened row-major format with color numbers by default.
    :param rng: Either `None`, or the `numpy.random.Generator` object that represents the random
        number generator used for all the probabilistic decisions. If this argument is `None`, then
        a default `numpy.random.Generator` object will be used. The default value is `None`.

    :return: The created `GraphGenerator` function.
    """

    # If the ``rng`` argument is `None`, then use a default `np.random.Generator`.
    if rng is None:
        rng = np.random.default_rng()  # pragma: no cover

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
        mask = rng.random(size=flattened_colors.shape) < edge_perturbation_probability
        entry_count = np.count_nonzero(mask)

        # If the argument ``color_selection_probabilities`` is `None`, just use the uniform
        # probability distribution.
        if color_selection_probabilities is None:
            flattened_colors[mask] = rng.integers(
                low=0, high=initial_graph.edge_colors, size=entry_count, dtype=np.uint8
            )
        # If the argument ``color_selection_probabilities`` is a `float`, then use this `float` as
        # the probability for coloring an edge (resp. arc) with the color 1.
        elif isinstance(color_selection_probabilities, float):
            flattened_colors[mask] = (
                rng.random(size=entry_count) < color_selection_probabilities
            ).astype(np.uint8)
        # If the argument ``color_selection_probabilities`` is a `np.ndarray`, then use the
        # `np.random.Generator.choice` method.
        else:
            flattened_colors[mask] = rng.choice(
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
    rng: Optional[np.random.Generator] = None,
) -> GraphGenerator:
    """
    This function creates a `GraphGenerator` function that outputs batches of fully colored
    $k$-edge-colored looped complete graphs where all the graphs are randomly generated. More
    precisely, each edge (resp. arc) from each graph is colored with a randomly chosen color
    between 0 and $k - 1$ with respect to a given discrete probability distribution. The graph
    order of all the graphs in the batch is configurable, as well as the number of proper edge
    colors in the generated batch. The user can also select whether the graphs in the batch should
    be directed or undirected, and whether they should be allowed to have loops. The output batch
    of graphs is initialized in one of the two flattened formats with color numbers, and the user
    can also configure which of these two formats should get used.

    :param graph_order: A positive `int` that represents the graph order of all the graphs in the
        batch that should be generated.
    :param color_selection_probabilities: Either a `numpy.ndarray` list of type `numpy.floating`, a
        `float` value, or `None`. This argument is used to describe the discrete probability
        distribution for coloring each of the edges (resp. arcs) from the graphs in the output
        batch. If the argument is a `numpy.ndarray` list, then the length of this list must be the
        same as the number of proper edge colors in the generated batch, i.e., the ``edge_colors``
        argument, and the list entries must be nonnegative numbers that sum up to 1. These entries
        are then used as the probabilities for each of the edge colors to be selected, in the
        ascending order. Alternatively, if the ``edge_colors`` argument is 2, then this argument
        can also be a `float` that represents the probability for the color 1 to be selected.
        Finally, the argument can also be `None`, in which case the uniform probability
        distribution is used, regardless of the value of the ``edge_colors`` argument. The default
        value is `None`.
    :param flattened_ordering: An item of the `FlattenedOrdering` enumeration that describes
        whether the flattened row-major format with color numbers or the flattened clockwise format
        with color numbers should be used to initialize the output batch of graphs. The default
        value is `FlattenedOrdering.ROW_MAJOR`, i.e., the output batch should get initialized in
        the flattened row-major format with color numbers by default.
    :param edge_colors: A positive `int` (not below 2) that represents the number of proper edge
        colors, i.e., $k$, in the output batch of graphs. The default value is 2.
    :param is_directed: A `bool` that indicates whether each of the graphs in the generated batch
        should be a $k$-edge-colored looped complete directed graph or a $k$-edge-colored looped
        complete undirected graph. The default value is `False`, i.e., the graphs are undirected by
        default.
    :param allow_loops: A `bool` that indicates whether each of the graphs in the generated batch
        should be allowed to have loops. The default value is `False`, i.e., the graphs are not
        allowed to have loops by default.
    :param rng: Either `None`, or the `numpy.random.Generator` object that represents the random
        number generator used for all the probabilistic decisions. If this argument is `None`, then
        a default `numpy.random.Generator` object will be used. The default value is `None`.

    :return: The created `GraphGenerator` function.
    """

    # If the ``rng`` argument is `None`, then use a default `np.random.Generator`.
    if rng is None:
        rng = np.random.default_rng()  # pragma: no cover

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
            flattened_colors = rng.integers(
                low=0, high=edge_colors, size=(batch_size, flattened_length), dtype=np.uint8
            )
        # If the argument ``color_selection_probabilities`` is a `float`, then use this `float` as
        # the probability for coloring an edge (resp. arc) with the color 1.
        elif isinstance(color_selection_probabilities, float):
            flattened_colors = (
                rng.random(size=(batch_size, flattened_length)) < color_selection_probabilities
            ).astype(np.uint8)
        # If the argument ``color_selection_probabilities`` is a `np.ndarray`, then use the
        # `np.random.Generator.choice` method.
        else:
            flattened_colors = rng.choice(
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
