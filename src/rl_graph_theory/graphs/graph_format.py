"""
This ``Python`` module contains the `GraphFormat` enumeration, which encapsulates the concept of a
format used to represent a $k$-edge-colored looped complete graph. The module additionally contains
the `BitmaskType` enumeration, which encapsulates the concept of a bitmask type in the context of
the two bitmask formats, and the `FlattenedOrdering` enumeration, which encapsulates the concept of
an edge (resp. arc) ordering in the context of the two flattened graph formats.
"""

from enum import Enum


class GraphFormat(Enum):
    r"""
    This enumeration encapsulates the concept of a format used to represent a $k$-edge-colored
    looped complete graph. Each graph is essentially represented as a quintuple ``(edge_colors,
    is_directed, allow_loops, graph_format, format_representation)``, where:

    * ``edge_colors`` is the number of proper edge colors, i.e., the value $k$;
    * ``is_directed`` is a boolean that indicates whether the considered graph is a
      $k$-edge-colored looped complete directed graph or a $k$-edge-colored looped complete
      undirected graph;
    * ``allow_loops`` is a boolean that indicates whether the considered graph is allowed to have
      loops (if loops are not allowed, then all the loops are removed from the considered looped
      complete graph and they simply do not exist);
    * ``graph_format`` is an item of this enumeration that describes the graph format that should
      be used to represent the structure of the considered graph; and
    * ``format_representation`` is a `numpy.ndarray` that represents the structure of the
      considered graph in the chosen graph format.

    The enumeration provides support for the following five graph formats:

    1. the bitmask format for the out-neighborhoods;
    2. the bitmask format for the in-neighborhoods;
    3. the adjacency matrix format;
    4. the flattened row-major format; and
    5. the flattened clockwise format.

    All of these formats require that the value $k$ be at most 255. Additionally, the two bitmask
    formats can only be used if the graph order is at most 64.

    :cvar BITMASK_OUT: The bitmask format for the out-neighborhoods. Here, the graph structure is
        represented through a `numpy.ndarray` matrix ``a`` of type `numpy.uint64` with $k$ rows and
        $n$ columns, where $n$ is the graph order. The value ``a[i, j]`` represents a nonnegative
        integer whose $k$-th bit indicates whether the edge (resp. arc) from the vertex $j$ to the
        vertex $k$ is of the color $i$. If loops are not allowed, then the $j$-bit of ``a[i, j]``
        is just zero. Additionally, if the graph is fully colored, then the starting row
        (corresponding to the color 0) of the said `numpy.ndarray` matrix is omitted, and we refer
        to this format as a reduced bitmask format.
    :cvar BITMASK_IN: The bitmask format for the in-neighborhoods. Here, the graph structure is
        represented through a `numpy.ndarray` matrix ``a`` of type `numpy.uint64` with $k$ rows and
        $n$ columns, where $n$ is the graph order. The value ``a[i, j]`` represents a nonnegative
        integer whose $k$-th bit indicates whether the edge (resp. arc) from the vertex $k$ to the
        vertex $j$ is of the color $i$. If loops are not allowed, then the $j$-bit of ``a[i, j]``
        is just zero. Additionally, if the graph is fully colored, then the starting row
        (corresponding to the color 0) of the said `numpy.ndarray` matrix is omitted, and we refer
        to this format as a reduced bitmask format.
    :cvar ADJACENCY_MATRIX: The adjacency matrix format. Here, the graph structure is represented
        through the adjacency matrix, i.e., a `numpy.ndarray` square matrix ``a`` of type
        `numpy.uint8` with $n$ rows and columns, where $n$ is the graph order. The value
        ``a[i, j]`` represents the color of the edge (resp. arc) from the vertex $i$ to the vertex
        $j$, with an uncolored edge (resp. arc) being represented by the value $k$. If loops are
        not allowed, then the diagonal entries of the adjacency matrix are all equal to zero.
    :cvar FLATTENED_ROW_MAJOR: The flattened row-major format. Here, the graph structure is
        represented through the adjacency matrix entries arranged in the row-major order. If the
        considered graph is directed and loops are not allowed, then the diagonal entries should be
        skipped when the entries are being arranged. If the considered graph is undirected, then
        only the entries from the upper triangular part of the adjacency matrix should be arranged
        in the row-major order (with or without the diagonal, depending on whether loops are
        allowed). The arranged entries form a `numpy.ndarray` list of type `numpy.uint8` of the
        required length.
    :cvar FLATTENED_CLOCKWISE: The flattened clockwise format. Here, the graph structure is
        represented through the adjacency matrix entries arranged in the clockwise layer order,
        i.e., the order $(0, 0), (0, 1), (1, 1), (1, 0), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0),
        (0, 3), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (3, 0), \\ldots$. If the considered graph
        is directed and loops are not allowed, then the diagonal entries should be skipped when the
        entries are being arranged. If the considered graph is undirected, then only the entries
        from the upper triangular part of the adjacency matrix should be arranged in the clockwise
        order (with or without the diagonal, depending on whether loops are allowed). The arranged
        entries form a `numpy.ndarray` list of type `numpy.uint8` of the required length.

    :note: If the graph is undirected, then the bitmask format for the out-neighborhoods and the
        bitmask format for the in-neighborhoods are the same. Also, if the graph is undirected,
        then the flattened clockwise format can be regarded as the flattened column-major format
        over the upper triangular part of the adjacency matrix (with or without the diagonal,
        depending on whether loops are allowed).
    """

    BITMASK_OUT = 0
    BITMASK_IN = 1
    ADJACENCY_MATRIX = 2
    FLATTENED_ROW_MAJOR = 3
    FLATTENED_CLOCKWISE = 4


class BitmaskType(Enum):
    """
    This enumeration encapsulates the concept of a bitmask type in the context of the two bitmask
    formats, `GraphFormat.BITMASK_OUT` and `GraphFormat.BITMASK_IN`, from the `GraphFormat`
    enumeration.

    :cvar OUT_NEIGHBORS: This item corresponds to the bitmask format for the out-neighborhoods.
    :cvar IN_NEIGHBORS: This item corresponds to the bitmask format for the in-neighborhoods.
    """

    OUT_NEIGHBORS = 0
    IN_NEIGHBORS = 1


class FlattenedOrdering(Enum):
    """
    This enumeration encapsulates the concept of an edge (resp. arc) ordering in the context of the
    two flattened graph formats, `GraphFormat.FLATTENED_ROW_MAJOR` and
    `GraphFormat.FLATTENED_CLOCKWISE`, from the `GraphFormat` enumeration.

    :cvar ROW_MAJOR: This item corresponds to the row-major edge (resp. arc) ordering.
    :cvar CLOCKWISE: This item corresponds to the clockwise edge (resp. arc) ordering.
    """

    ROW_MAJOR = 0
    CLOCKWISE = 1
