"""
The `rl_graph_theory.graphs` package contains the `Graph` class that encapsulates the concept of a
$k$-edge-colored looped complete graph and a batch of $k$-edge-colored looped complete graphs of
the same order. The package also contains the `GraphFormat` enumeration, which encapsulates the
concept of a format used to represent a $k$-edge-colored looped complete graph, alongside several
auxiliary functions and various classes that are used to construct $k$-edge-colored looped complete
graphs with some particular structure.
"""

from .graph import Graph
from .graph_formats import (
    BitmaskType,
    ColorRepresentation,
    FlattenedOrdering,
    GraphFormat,
)
from .special_graphs import (
    AlmostCompleteGraph,
    BookGraph,
    CompleteBipartiteGraph,
    CompleteGraph,
    CompleteKPartiteGraph,
    CycleGraph,
    EmptyGraph,
    FriendshipGraph,
    MonochromaticGraph,
    PathGraph,
    StarGraph,
    WheelGraph,
)
from .utils import (
    binary_slices_to_color_numbers,
    color_numbers_to_binary_slices,
    compute_edge_indices,
    flatten_from_adjacency_matrix,
    flattened_length_to_graph_order,
    graph_order_to_flattened_length,
    unflatten_to_adjacency_matrix,
)


__all__ = [
    "Graph",
    "BitmaskType",
    "ColorRepresentation",
    "FlattenedOrdering",
    "GraphFormat",
    "AlmostCompleteGraph",
    "BookGraph",
    "CompleteBipartiteGraph",
    "CompleteGraph",
    "CompleteKPartiteGraph",
    "CycleGraph",
    "EmptyGraph",
    "FriendshipGraph",
    "MonochromaticGraph",
    "PathGraph",
    "StarGraph",
    "WheelGraph",
    "binary_slices_to_color_numbers",
    "color_numbers_to_binary_slices",
    "compute_edge_indices",
    "flatten_from_adjacency_matrix",
    "flattened_length_to_graph_order",
    "graph_order_to_flattened_length",
    "unflatten_to_adjacency_matrix",
]
