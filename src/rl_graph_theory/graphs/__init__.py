"""
The `rl_graph_theory.graphs` package contains the two core classes `Graph` and `GraphBatch`, which
encapsulate the concept of an edge-colored complete graph and a batch of edge-colored complete
graphs of the same order, respectively, alongside various other classes that are used to construct
edge-colored complete graphs with some particular structure.
"""

from .graph import EdgeOrdering, Graph, GraphBatch, GraphFormat
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


__all__ = [
    "EdgeOrdering",
    "Graph",
    "GraphBatch",
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
]
