"""
This ``Python`` module contains the `Graph` class, which encapsulates the concept of a
$k$-edge-colored looped complete graph or a batch of $k$-edge-colored looped complete graphs of the
same order.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .graph_formats import (
    BitmaskType,
    ColorRepresentation,
    FlattenedOrdering,
    GraphFormat,
)
from .utils import (
    binary_slices_to_color_numbers,
    color_numbers_to_binary_slices,
    flatten_from_adjacency_matrix,
    flattened_length_to_graph_order,
    unflatten_to_adjacency_matrix,
)


class Graph:
    r"""
    This class encapsulates the concept of a $k$-edge-colored looped complete graph, or a batch of
    $k$-edge-colored looped complete graphs of the same order, with $k$ being at least 2. If a
    single graph is being modeled, then it is represented as a quintuple ``(edge_colors,
    is_directed, allow_loops, graph_format, format_representation)``, as explained in the
    `GraphFormat` enumeration. If a batch of graphs is being modeled, then the graphs in the batch
    must be consistent: they should all have the same number of proper edge colors and the same
    order, they should all be directed or all be undirected, and they should all either allow loops
    or not allow loops. Therefore, a batch of graphs can be analogously represented through a
    quintuple ``(edge_colors, is_directed, allow_loops, graph_format, format_representation)``. In
    this case, the only difference is that all of the `numpy.ndarray` objects to be used as format
    representations should be of one dimension higher. More precisely, if ``a`` is the
    `numpy.ndarray` used to describe the structure of a given batch of graphs in some graph format,
    then ``a[i]`` is the structural representation of the $i$-th graph in the batch in the
    corresponding format.

    The structure of the graph (resp. batch of graphs) can be represented in any of the 8 graph
    formats from the `GraphFormat` enumeration. It is initialized in at least one of these 8
    formats so that the ``format_representation`` `numpy.ndarray` objects representing the graph
    structure are internally stored. Afterwards, the graph (resp. batch of graphs) can be accessed
    in any of the 8 formats, with all the required graph format conversions being performed
    automatically. With each performed conversion, the obtained `numpy.ndarray` object
    corresponding to the output graph format is internally stored. The class provides support for
    all the reduced formats, and while performing a conversion to a bitmask format or one of the
    formats with binary slices, the corresponding reduced format is always used if possible, i.e.,
    if the graph (resp. batch of graphs) is fully colored.

    :ivar __edge_colors: The number of proper edge colors, i.e., $k$, given as a positive `int`
        that is at least 2.
    :ivar __is_directed: A `bool` that indicates whether the given graph (resp. each graph in the
        given batch) is a $k$-edge-colored looped complete directed graph or a $k$-edge-colored
        looped complete undirected graph.
    :ivar __allow_loops: A `bool` that indicates whether the given graph (resp. each graph in the
        given batch) is allowed to have loops. If loops are not allowed, then all the loops are
        removed from the considered looped complete graph(s) and they simply do not exist.
    :ivar __batch_size: If the object models a single graph, then `None`, and if the object models
        a batch of graphs, then a positive `int` that determines the batch size, i.e., the number
        of graphs in the batch.
    :ivar __graph_order: The graph order (resp. the common order of each of the graphs in the
        batch), given as a positve `int`.
    :ivar __bitmask_out: The `numpy.ndarray` of type `numpy.uint64` that represents the graph
        structure in the bitmask format for the out-neighborhoods, if it was used while
        initializing the graph (resp. batch of graphs) or computed afterwards, and otherwise,
        `None`.
    :ivar __bitmask_in: The `numpy.ndarray` of type `numpy.uint64` that represents the graph
        structure in the bitmask format for the in-neighborhoods, if it was used while initializing
        the graph (resp. batch of graphs) or computed afterwards, and otherwise, `None`.
    :ivar __adjacency_matrix_colors: The `numpy.ndarray` of type `numpy.uint8` that represents the
        graph structure in the adjacency matrix format with color numbers, if it was used while
        initializing the graph (resp. batch of graphs) or computed afterwards, and otherwise,
        `None`.
    :ivar __adjacency_matrix_binary: The `numpy.ndarray` of type `numpy.uint8` that represents the
        graph structure in the adjacency matrix format with binary slices, if it was used while
        initializing the graph (resp. batch of graphs) or computed afterwards, and otherwise,
        `None`.
    :ivar __flattened_row_major_colors: The `numpy.ndarray` of type `numpy.uint8` that represents
        the graph structure in the flattened row-major format with color numbers, if it was used
        while initializing the graph (resp. batch of graphs) or computed afterwards, and otherwise,
        `None`.
    :ivar __flattened_row_major_binary: The `numpy.ndarray` of type `numpy.uint8` that represents
        the graph structure in the flattened row-major format with binary slices, if it was used
        while initializing the graph (resp. batch of graphs) or computed afterwards, and otherwise,
        `None`.
    :ivar __flattened_clockwise_colors: The `numpy.ndarray` of type `numpy.uint8` that represents
        the graph structure in the flattened clockwise format with color numbers, if it was used
        while initializing the graph (resp. batch of graphs) or computed afterwards, and otherwise,
        `None`.
    :ivar __flattened_clockwise_binary: The `numpy.ndarray` of type `numpy.uint8` that represents
        the graph structure in the flattened clockwise format with binary slices, if it was used
        while initializing the graph (resp. batch of graphs) or computed afterwards, and otherwise,
        `None`.
    """

    def __init__(
        self,
        edge_colors: int = 2,
        is_directed: bool = False,
        allow_loops: bool = False,
        bitmask_out: Optional[np.ndarray] = None,
        bitmask_in: Optional[np.ndarray] = None,
        adjacency_matrix_colors: Optional[np.ndarray] = None,
        adjacency_matrix_binary: Optional[np.ndarray] = None,
        flattened_row_major_colors: Optional[np.ndarray] = None,
        flattened_row_major_binary: Optional[np.ndarray] = None,
        flattened_clockwise_colors: Optional[np.ndarray] = None,
        flattened_clockwise_binary: Optional[np.ndarray] = None,
    ):
        """
        This constructor initializes an instance, which models either a graph or a batch of graphs,
        in at least one of the 8 graph formats from the `GraphFormat` enumeration. In other words,
        at least one of the 8 graph formats is selected, and then the graph (resp. batch of graphs)
        is initialized in each of the selected formats using the corresponding provided
        `numpy.ndarray` objects. If the instance is initialized in more than one graph format, then
        the provided `numpy.ndarray` objects must be consistent with one another, i.e., they must
        all represent the same graph or batch of graphs.

        :param edge_colors: A positive `int` (not below 2) that represents the number of proper
            edge colors, i.e., $k$. The default value is 2.
        :param is_directed: A `bool` that indicates whether the given graph (resp. each graph in
            the given batch) is a $k$-edge-colored looped complete directed graph or a
            $k$-edge-colored looped complete undirected graph. The default value is `False`, i.e.,
            the given graph (resp. each graph in the given batch) is undirected by default.
        :param allow_loops: A `bool` that indicates whether the given graph (resp. each graph in
            the given batch) is allowed to have loops. If loops are not allowed, then all the loops
            are removed from the considered looped complete graph(s) and they simply do not exist.
            The default value is `False`, i.e., the given graph (resp. each graph in the given
            batch) is not allowed to have loops by default.
        :param bitmask_out: If the graph (resp. batch of graphs) should be initialized in the
            bitmask format for the out-neighborhoods, then this argument should be the
            `numpy.ndarray` of type `numpy.uint64` that describes the format representation in this
            graph format, and otherwise, it should be `None`.
        :param bitmask_in: If the graph (resp. batch of graphs) should be initialized in the
            bitmask format for the in-neighborhoods, then this argument should be the
            `numpy.ndarray` of type `numpy.uint64` that describes the format representation in this
            graph format, and otherwise, it should be `None`.
        :param adjacency_matrix_colors: If the graph (resp. batch of graphs) should be initialized
            in the adjacency matrix format with color numbers, then this argument should be the
            `numpy.ndarray` of type `numpy.uint8` that describes the format representation in this
            graph format, and otherwise, it should be `None`.
        :param adjacency_matrix_binary: If the graph (resp. batch of graphs) should be initialized
            in the adjacency matrix format with binary slices, then this argument should be the
            `numpy.ndarray` of type `numpy.uint8` that describes the format representation in this
            graph format, and otherwise, it should be `None`.
        :param flattened_row_major_colors: If the graph (resp. batch of graphs) should be
            initialized in the flattened row-major format with color numbers, then this argument
            should be the `numpy.ndarray` of type `numpy.uint8` that describes the format
            representation in this graph format, and otherwise, it should be `None`.
        :param flattened_row_major_binary: If the graph (resp. batch of graphs) should be
            initialized in the flattened row-major format with binary slices, then this argument
            should be the `numpy.ndarray` of type `numpy.uint8` that describes the format
            representation in this graph format, and otherwise, it should be `None`.
        :param flattened_clockwise_colors: If the graph (resp. batch of graphs) should be
            initialized in the flattened clockwise format with color numbers, then this argument
            should be the `numpy.ndarray` of type `numpy.uint8` that describes the format
            representation in this graph format, and otherwise, it should be `None`.
        :param flattened_clockwise_binary: If the graph (resp. batch of graphs) should be
            initialized in the flattened clockwise format with binary slices, then this argument
            should be the `numpy.ndarray` of type `numpy.uint8` that describes the format
            representation in this graph format, and otherwise, it should be `None`.
        """

        self.__edge_colors: int = edge_colors
        self.__is_directed: bool = is_directed
        self.__allow_loops: bool = allow_loops

        # Initialize the graph (resp. batch of graphs) in all the selected graph formats.
        self.__bitmask_out: Optional[np.ndarray] = bitmask_out
        self.__bitmask_in: Optional[np.ndarray] = bitmask_in
        self.__adjacency_matrix_colors: Optional[np.ndarray] = adjacency_matrix_colors
        self.__adjacency_matrix_binary: Optional[np.ndarray] = adjacency_matrix_binary
        self.__flattened_row_major_colors: Optional[np.ndarray] = flattened_row_major_colors
        self.__flattened_row_major_binary: Optional[np.ndarray] = flattened_row_major_binary
        self.__flattened_clockwise_colors: Optional[np.ndarray] = flattened_clockwise_colors
        self.__flattened_clockwise_binary: Optional[np.ndarray] = flattened_clockwise_binary

        # If the graph (resp. each graph from the batch) is undirected, then the two bitmask
        # formats are the same.
        if not is_directed:
            if bitmask_out is not None:
                self.__bitmask_in = bitmask_out
            elif bitmask_in is not None:
                self.__bitmask_out = bitmask_in

        # Determine the batch size (potentially `None`) and the graph order using the bitmask
        # format for the out-neighborhoods, if possible.
        if bitmask_out is not None:
            self.__batch_size: Optional[int] = (
                None if len(bitmask_out.shape) == 2 else bitmask_out.shape[0]
            )
            self.__graph_order: int = bitmask_out.shape[-1]

        # Otherwise, determine the batch size (potentially `None`) and the graph order using the
        # bitmask format for the in-neighborhoods, if possible.
        elif bitmask_in is not None:
            self.__batch_size: Optional[int] = (
                None if len(bitmask_in.shape) == 2 else bitmask_in.shape[0]
            )
            self.__graph_order: int = bitmask_in.shape[-1]

        # Otherwise, determine the batch size (potentially `None`) and the graph order using the
        # adjacency matrix format with color numbers, if possible.
        elif adjacency_matrix_colors is not None:
            self.__batch_size: Optional[int] = (
                None
                if len(adjacency_matrix_colors.shape) == 2
                else adjacency_matrix_colors.shape[0]
            )
            self.__graph_order: int = adjacency_matrix_colors.shape[-1]

        # Otherwise, determine the batch size (potentially `None`) and the graph order using the
        # adjacency matrix format with binary slices, if possible.
        elif adjacency_matrix_binary is not None:
            self.__batch_size: Optional[int] = (
                None
                if len(adjacency_matrix_binary.shape) == 3
                else adjacency_matrix_binary.shape[0]
            )
            self.__graph_order: int = adjacency_matrix_binary.shape[-1]

        # Otherwise, determine the batch size (potentially `None`) and the graph order using the
        # flattened row-major format with color numbers, if possible.
        elif flattened_row_major_colors is not None:
            self.__batch_size: Optional[int] = (
                None
                if len(flattened_row_major_colors.shape) == 1
                else flattened_row_major_colors.shape[0]
            )
            self.__graph_order: int = flattened_length_to_graph_order(
                flattened_length=flattened_row_major_colors.shape[-1],
                is_directed=is_directed,
                allow_loops=allow_loops,
            )

        # Otherwise, determine the batch size (potentially `None`) and the graph order using the
        # flattened row-major format with binary slices, if possible.
        elif flattened_row_major_binary is not None:
            self.__batch_size: Optional[int] = (
                None
                if len(flattened_row_major_binary.shape) == 2
                else flattened_row_major_binary.shape[0]
            )
            self.__graph_order: int = flattened_length_to_graph_order(
                flattened_length=flattened_row_major_binary.shape[-1],
                is_directed=is_directed,
                allow_loops=allow_loops,
            )

        # Otherwise, determine the batch size (potentially `None`) and the graph order using the
        # flattened clockwise format with color numbers, if possible.
        elif flattened_clockwise_colors is not None:
            self.__batch_size: Optional[int] = (
                None
                if len(flattened_clockwise_colors.shape) == 1
                else flattened_clockwise_colors.shape[0]
            )
            self.__graph_order: int = flattened_length_to_graph_order(
                flattened_length=flattened_clockwise_colors.shape[-1],
                is_directed=is_directed,
                allow_loops=allow_loops,
            )

        # Otherwise, determine the batch size (potentially `None`) and the graph order using the
        # flattened clockwise format with binary slices.
        else:
            self.__batch_size: Optional[int] = (
                None
                if len(flattened_clockwise_binary.shape) == 2
                else flattened_clockwise_binary.shape[0]
            )
            self.__graph_order: int = flattened_length_to_graph_order(
                flattened_length=flattened_clockwise_binary.shape[-1],
                is_directed=is_directed,
                allow_loops=allow_loops,
            )

    @classmethod
    def from_bitmask(
        cls,
        bitmask: np.ndarray,
        bitmask_type: BitmaskType = BitmaskType.OUT_NEIGHBORS,
        edge_colors: int = 2,
        is_directed: bool = False,
        allow_loops: bool = False,
    ) -> Graph:
        """
        This class method initializes a `Graph` object in either the bitmask format for the
        out-neighborhoods or the bitmask format for the in-neighborhoods.

        :param bitmask: A `numpy.ndarray` of type `numpy.uint64` that describes the format
            representation of the graph (resp. batch of graphs) that should be initialized, either
            in the bitmask format for the out-neighborhoods or the bitmask format for the
            in-neighborhoods.
        :param bitmask_type: An item of the `BitmaskType` enumeration that determines whether the
            graph should be initialized in the bitmask format for the out-neighborhoods or the
            bitmask format for the in-neighborhoods. The default value is
            `BitmaskType.OUT_NEIGHBORS`, i.e., the graph should be initialized in the bitmask
            format for the out-neighborhoods by default.
        :param edge_colors: A positive `int` (not below 2) that represents the number of proper
            edge colors, i.e., $k$. The default value is 2.
        :param is_directed: A `bool` that indicates whether the given graph (resp. each graph in
            the given batch) is a $k$-edge-colored looped complete directed graph or a
            $k$-edge-colored looped complete undirected graph. The default value is `False`, i.e.,
            the given graph (resp. each graph in the given batch) is undirected by default.
        :param allow_loops: A `bool` that indicates whether the given graph (resp. each graph in
            the given batch) is allowed to have loops. If loops are not allowed, then all the loops
            are removed from the considered looped complete graph(s) and they simply do not exist.
            The default value is `False`, i.e., the given graph (resp. each graph in the given
            batch) is not allowed to have loops by default.

        :return: The initialized `Graph` object.

        :note: The class method provides support for the reduced bitmask formats, i.e., the graph
            (resp. batch of graphs) can also be initialized in any of the two reduced bitmask
            formats.
        """

        if bitmask_type == BitmaskType.OUT_NEIGHBORS:
            return cls(
                edge_colors=edge_colors,
                is_directed=is_directed,
                allow_loops=allow_loops,
                bitmask_out=bitmask,
            )
        else:
            return cls(
                edge_colors=edge_colors,
                is_directed=is_directed,
                allow_loops=allow_loops,
                bitmask_in=bitmask,
            )

    @classmethod
    def from_adjacency_matrix(
        cls,
        adjacency_matrix: np.ndarray,
        color_representation: ColorRepresentation = ColorRepresentation.COLOR_NUMBERS,
        edge_colors: int = 2,
        is_directed: bool = False,
        allow_loops: bool = False,
    ) -> Graph:
        """
        This class method initializes a `Graph` object in either the adjacency matrix format with
        color numbers or the adjacency matrix format with binary slices.

        :param adjacency_matrix: A `numpy.ndarray` of type `numpy.uint8` that describes the format
            representation of the graph (resp. batch of graphs) that should be initialized, either
            in the adjacency matrix format with color numbers or the adjacency matrix format with
            binary slices.
        :param color_representation: An item of the `ColorRepresentation` enumeration that
            determines whether the graph should be initialized in the adjacency matrix format with
            color numbers or the adjacency matrix format with binary slices. The default value is
            `ColorRepresentation.COLOR_NUMBERS`, i.e., the graph should be initialized in the
            adjacency matrix format with color numbers by default.
        :param edge_colors: A positive `int` (not below 2) that represents the number of proper
            edge colors, i.e., $k$. The default value is 2.
        :param is_directed: A `bool` that indicates whether the given graph (resp. each graph in
            the given batch) is a $k$-edge-colored looped complete directed graph or a
            $k$-edge-colored looped complete undirected graph. The default value is `False`, i.e.,
            the given graph (resp. each graph in the given batch) is undirected by default.
        :param allow_loops: A `bool` that indicates whether the given graph (resp. each graph in
            the given batch) is allowed to have loops. If loops are not allowed, then all the loops
            are removed from the considered looped complete graph(s) and they simply do not exist.
            The default value is `False`, i.e., the given graph (resp. each graph in the given
            batch) is not allowed to have loops by default.

        :return: The initialized `Graph` object.

        :note: The class method provides support for the reduced adjacency matrix format with
            binary slices, i.e., if the graph (resp. batch of graphs) should be initialized in this
            format, then the reduced variant is also accepted.
        """

        if color_representation == ColorRepresentation.COLOR_NUMBERS:
            return cls(
                edge_colors=edge_colors,
                is_directed=is_directed,
                allow_loops=allow_loops,
                adjacency_matrix_colors=adjacency_matrix,
            )
        else:
            return cls(
                edge_colors=edge_colors,
                is_directed=is_directed,
                allow_loops=allow_loops,
                adjacency_matrix_binary=adjacency_matrix,
            )

    @classmethod
    def from_flattened(
        cls,
        flattened: np.ndarray,
        flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
        color_representation: ColorRepresentation = ColorRepresentation.COLOR_NUMBERS,
        edge_colors: int = 2,
        is_directed: bool = False,
        allow_loops: bool = False,
    ) -> Graph:
        """
        This class method initializes a `Graph` object in exactly one of the four possible
        flattened formats.

        :param flattened: A `numpy.ndarray` of type `numpy.uint8` that describes the format
            representation of the graph (resp. batch of graphs) that should be initialized, in one
            of the four possible flattened formats.
        :param flattened_ordering: An item of the `FlattenedOrdering` enumeration that determines
            whether the graph should be initialized in a flattened row-major format or a flattened
            clockwise format. The default value is `FlattenedOrdering.ROW_MAJOR`, i.e., the graph
            should be initialized in a flattened row-major format by default.
        :param color_representation: An item of the `ColorRepresentation` enumeration that
            determines whether the graph should be initialized in a flattened format with color
            numbers or a flattened format with binary slices. The default value is
            `ColorRepresentation.COLOR_NUMBERS`, i.e., the graph should be initialized in a
            flattened format with color numbers by default.
        :param edge_colors: A positive `int` (not below 2) that represents the number of proper
            edge colors, i.e., $k$. The default value is 2.
        :param is_directed: A `bool` that indicates whether the given graph (resp. each graph in
            the given batch) is a $k$-edge-colored looped complete directed graph or a
            $k$-edge-colored looped complete undirected graph. The default value is `False`, i.e.,
            the given graph (resp. each graph in the given batch) is undirected by default.
        :param allow_loops: A `bool` that indicates whether the given graph (resp. each graph in
            the given batch) is allowed to have loops. If loops are not allowed, then all the loops
            are removed from the considered looped complete graph(s) and they simply do not exist.
            The default value is `False`, i.e., the given graph (resp. each graph in the given
            batch) is not allowed to have loops by default.

        :return: The initialized `Graph` object.

        :note: The class method provides support for the reduced flattened formats with binary
            slices, i.e., if the graph (resp. batch of graphs) should be initialized in one of
            these two formats, then the reduced variants are also accepted.
        """

        if flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            if color_representation == ColorRepresentation.COLOR_NUMBERS:
                return cls(
                    edge_colors=edge_colors,
                    is_directed=is_directed,
                    allow_loops=allow_loops,
                    flattened_row_major_colors=flattened,
                )
            else:
                return cls(
                    edge_colors=edge_colors,
                    is_directed=is_directed,
                    allow_loops=allow_loops,
                    flattened_row_major_binary=flattened,
                )
        else:
            if color_representation == ColorRepresentation.COLOR_NUMBERS:
                return cls(
                    edge_colors=edge_colors,
                    is_directed=is_directed,
                    allow_loops=allow_loops,
                    flattened_clockwise_colors=flattened,
                )
            else:
                return cls(
                    edge_colors=edge_colors,
                    is_directed=is_directed,
                    allow_loops=allow_loops,
                    flattened_clockwise_binary=flattened,
                )

    @property
    def edge_colors(self) -> int:
        """
        This property returns the number of proper edge colors in the given graph (resp. batch of
        graphs), i.e., $k$, as a positive `int` that is at least 2.
        """

        return self.__edge_colors

    @property
    def is_directed(self) -> bool:
        """
        This property returns the `bool` that determines whether the given graph (resp. each graph
        in the given batch) is directed. The value `True` indicates that the graph (resp. each
        graph) is directed.
        """

        return self.__is_directed

    @property
    def allow_loops(self) -> bool:
        """
        This property returns the `bool` that determines whether loops are allowed in the given
        graph (resp. each graph in the given batch). The value `True` indicates that loops are
        allowed.
        """

        return self.__allow_loops

    @property
    def batch_size(self) -> Optional[int]:
        """
        In the case of a $k$-edge-colored looped complete graph, this property returns `None`. In
        the case of a batch of $k$-edge-colored looped complete graphs of the same order, this
        property returns the batch size, i.e., the number of graphs in the batch, as a positive
        `int`.
        """

        return self.__batch_size

    @property
    def graph_order(self) -> int:
        """
        This property returns the order of the given graph (resp. the common order of all the
        graphs in the given batch), as a positive `int`.
        """

        return self.__graph_order

    def __convert_graph_format(
        self, input_format: GraphFormat, output_format: GraphFormat
    ) -> None:
        """
        This private method performs a direct conversion from the format representation of the
        given graph (resp. batch of graphs) in a selected input format to the format representation
        of that graph (resp. batch of graphs) in a selected output format. The following direct
        conversions are supported:

        1. bitmask format for the out-neighborhoods ↔ adjacency matrix format with binary slices;
        2. bitmask format for the in-neighborhoods ↔ adjacency matrix format with binary slices;
        3. adjacency matrix format with binary slices ↔ flattened row-major format with binary
           slices;
        4. adjacency matrix format with binary slices ↔ flattened clockwise format with binary
           slices;
        5. adjacency matrix format with color numbers ↔ flattened row-major format with color
           numbers;
        6. adjacency matrix format with color numbers ↔ flattened clockwise format with color
           numbers;
        7. adjacency matrix format with binary slices ↔ adjacency matrix format with color numbers;
        8. flattened row-major format with binary slices ↔ flattened row-major format with color
           numbers; and
        9. flattened clockwise format with binary slices ↔ flattened clockwise format with color
           numbers.

        A `NotImplementedError` is raised if any other direct conversion is attempted.

        :param input_format: The selected input format, given as an item of the `GraphFormat`
            enumeration.
        :param output_format: The selected output format, given as an item of the `GraphFormat`
            enumeration.
        """

        # The bitmask format for the out-neighborhoods can be directly converted only to the
        # adjacency matrix format with binary slices.
        if input_format == GraphFormat.BITMASK_OUT:
            if output_format == GraphFormat.ADJACENCY_MATRIX_BINARY:
                masks = 1 << np.arange(self.__graph_order, dtype=np.uint64)
                temp = np.expand_dims(self.__bitmask_out, axis=-1) & masks
                self.__adjacency_matrix_binary = (temp != 0).astype(np.uint8)
            else:
                raise NotImplementedError

        # The bitmask format for the in-neighborhoods can be directly converted only to the
        # adjacency matrix format with binary slices.
        elif input_format == GraphFormat.BITMASK_IN:
            if output_format == GraphFormat.ADJACENCY_MATRIX_BINARY:
                masks = 1 << np.arange(self.__graph_order, dtype=np.uint64)
                temp = np.expand_dims(self.__bitmask_in, axis=-1) & masks
                temp = (temp != 0).astype(np.uint8)
                self.__adjacency_matrix_binary = np.swapaxes(temp, -1, -2)
            else:
                raise NotImplementedError

        # The adjacency matrix format with binary slices can be directly converted to any of the
        # two bitmask formats, to the adjacency matrix format with color numbers, and to any of the
        # two flattened formats with binary slices.
        elif input_format == GraphFormat.ADJACENCY_MATRIX_BINARY:
            if output_format == GraphFormat.BITMASK_OUT:
                masks = 1 << np.arange(self.__graph_order, dtype=np.uint64)
                self.__bitmask_out = self.__adjacency_matrix_binary @ masks
            elif output_format == GraphFormat.BITMASK_IN:
                masks = 1 << np.arange(self.__graph_order, dtype=np.uint64)
                self.__bitmask_in = np.tensordot(
                    masks, self.__adjacency_matrix_binary, axes=(0, -2)
                )
            elif output_format == GraphFormat.ADJACENCY_MATRIX_COLORS:
                self.__adjacency_matrix_colors = binary_slices_to_color_numbers(
                    input_representation=self.__adjacency_matrix_binary,
                    is_flattened_format=False,
                    edge_colors=self.__edge_colors,
                    allow_loops=self.__allow_loops,
                )
            elif output_format == GraphFormat.FLATTENED_ROW_MAJOR_BINARY:
                self.__flattened_row_major_binary = flatten_from_adjacency_matrix(
                    adjacency_matrix=self.__adjacency_matrix_binary,
                    flattened_ordering=FlattenedOrdering.ROW_MAJOR,
                    is_directed=self.__is_directed,
                    allow_loops=self.__allow_loops,
                )
            elif output_format == GraphFormat.FLATTENED_CLOCKWISE_BINARY:
                self.__flattened_clockwise_binary = flatten_from_adjacency_matrix(
                    adjacency_matrix=self.__adjacency_matrix_binary,
                    flattened_ordering=FlattenedOrdering.CLOCKWISE,
                    is_directed=self.__is_directed,
                    allow_loops=self.__allow_loops,
                )
            else:
                raise NotImplementedError

        # The adjacency matrix format with color numbers can be directly converted to the adjacency
        # matrix format with binary slices, and to any of the two flattened formats with color
        # numbers.
        elif input_format == GraphFormat.ADJACENCY_MATRIX_COLORS:
            if output_format == GraphFormat.ADJACENCY_MATRIX_BINARY:
                self.__adjacency_matrix_binary = color_numbers_to_binary_slices(
                    input_representation=self.__adjacency_matrix_colors,
                    is_flattened_format=False,
                    edge_colors=self.__edge_colors,
                    allow_loops=self.__allow_loops,
                )
            elif output_format == GraphFormat.FLATTENED_ROW_MAJOR_COLORS:
                self.__flattened_row_major_colors = flatten_from_adjacency_matrix(
                    adjacency_matrix=self.__adjacency_matrix_colors,
                    flattened_ordering=FlattenedOrdering.ROW_MAJOR,
                    is_directed=self.__is_directed,
                    allow_loops=self.__allow_loops,
                )
            elif output_format == GraphFormat.FLATTENED_CLOCKWISE_COLORS:
                self.__flattened_clockwise_colors = flatten_from_adjacency_matrix(
                    adjacency_matrix=self.__adjacency_matrix_colors,
                    flattened_ordering=FlattenedOrdering.CLOCKWISE,
                    is_directed=self.__is_directed,
                    allow_loops=self.__allow_loops,
                )
            else:
                raise NotImplementedError

        # The flattened row-major format with binary slices can be directly converted only to the
        # adjacency matrix format with binary slices and the flattened row-major format with color
        # numbers.
        elif input_format == GraphFormat.FLATTENED_ROW_MAJOR_BINARY:
            if output_format == GraphFormat.ADJACENCY_MATRIX_BINARY:
                self.__adjacency_matrix_binary = unflatten_to_adjacency_matrix(
                    flattened=self.__flattened_row_major_binary,
                    flattened_ordering=FlattenedOrdering.ROW_MAJOR,
                    is_directed=self.__is_directed,
                    allow_loops=self.__allow_loops,
                    graph_order=self.__graph_order,
                )
            elif output_format == GraphFormat.FLATTENED_ROW_MAJOR_COLORS:
                self.__flattened_row_major_colors = binary_slices_to_color_numbers(
                    input_representation=self.__flattened_row_major_binary,
                    is_flattened_format=True,
                    edge_colors=self.__edge_colors,
                    allow_loops=self.__allow_loops,
                )
            else:
                raise NotImplementedError

        # The flattened clockwise format with binary slices can be directly converted only to the
        # adjacency matrix format with binary slices and the flattened clockwise format with color
        # numbers.
        elif input_format == GraphFormat.FLATTENED_CLOCKWISE_BINARY:
            if output_format == GraphFormat.ADJACENCY_MATRIX_BINARY:
                self.__adjacency_matrix_binary = unflatten_to_adjacency_matrix(
                    flattened=self.__flattened_clockwise_binary,
                    flattened_ordering=FlattenedOrdering.CLOCKWISE,
                    is_directed=self.__is_directed,
                    allow_loops=self.__allow_loops,
                    graph_order=self.__graph_order,
                )
            elif output_format == GraphFormat.FLATTENED_CLOCKWISE_COLORS:
                self.__flattened_clockwise_colors = binary_slices_to_color_numbers(
                    input_representation=self.__flattened_clockwise_binary,
                    is_flattened_format=True,
                    edge_colors=self.__edge_colors,
                    allow_loops=self.__allow_loops,
                )
            else:
                raise NotImplementedError

        # The flattened row-major format with color numbers can be directly converted only to the
        # adjacency matrix format with color numbers and the flattened row-major format with binary
        # slices.
        elif input_format == GraphFormat.FLATTENED_ROW_MAJOR_COLORS:
            if output_format == GraphFormat.ADJACENCY_MATRIX_COLORS:
                self.__adjacency_matrix_colors = unflatten_to_adjacency_matrix(
                    flattened=self.__flattened_row_major_colors,
                    flattened_ordering=FlattenedOrdering.ROW_MAJOR,
                    is_directed=self.__is_directed,
                    allow_loops=self.__allow_loops,
                    graph_order=self.__graph_order,
                )
            elif output_format == GraphFormat.FLATTENED_ROW_MAJOR_BINARY:
                self.__flattened_row_major_binary = color_numbers_to_binary_slices(
                    input_representation=self.__flattened_row_major_colors,
                    is_flattened_format=True,
                    edge_colors=self.__edge_colors,
                    allow_loops=self.__allow_loops,
                )
            else:
                raise NotImplementedError

        # The flattened clockwise format with color numbers can be directly converted only to the
        # adjacency matrix format with color numbers and the flattened clockwise format with binary
        # slices.
        elif input_format == GraphFormat.FLATTENED_CLOCKWISE_COLORS:
            if output_format == GraphFormat.ADJACENCY_MATRIX_COLORS:
                self.__adjacency_matrix_colors = unflatten_to_adjacency_matrix(
                    flattened=self.__flattened_clockwise_colors,
                    flattened_ordering=FlattenedOrdering.CLOCKWISE,
                    is_directed=self.__is_directed,
                    allow_loops=self.__allow_loops,
                    graph_order=self.__graph_order,
                )
            elif output_format == GraphFormat.FLATTENED_CLOCKWISE_BINARY:
                self.__flattened_clockwise_binary = color_numbers_to_binary_slices(
                    input_representation=self.__flattened_clockwise_colors,
                    is_flattened_format=True,
                    edge_colors=self.__edge_colors,
                    allow_loops=self.__allow_loops,
                )
            else:
                raise NotImplementedError

    @property
    def bitmask_out(self) -> np.ndarray:
        """
        This property returns the `numpy.ndarray` of type `numpy.uint64` that describes the format
        representation of the given graph (resp. batch of graphs) in the bitmask format for the
        out-neighborhoods.
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__bitmask_out is not None:
            return self.__bitmask_out

        # Otherwise, determine the output `np.ndarray` by relying on the adjacency matrix format
        # with binary slices.
        self.adjacency_matrix_binary
        self.__convert_graph_format(
            input_format=GraphFormat.ADJACENCY_MATRIX_BINARY, output_format=GraphFormat.BITMASK_OUT
        )

        return self.__bitmask_out

    @property
    def bitmask_in(self) -> np.ndarray:
        """
        This property returns the `numpy.ndarray` of type `numpy.uint64` that describes the format
        representation of the given graph (resp. batch of graphs) in the bitmask format for the
        in-neighborhoods.
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__bitmask_in is not None:
            return self.__bitmask_in

        # Otherwise, determine the output `np.ndarray` by relying on the adjacency matrix format
        # with binary slices.
        self.adjacency_matrix_binary
        self.__convert_graph_format(
            input_format=GraphFormat.ADJACENCY_MATRIX_BINARY, output_format=GraphFormat.BITMASK_IN
        )

        return self.__bitmask_in

    @property
    def adjacency_matrix_colors(self) -> np.ndarray:
        """
        This property returns the `numpy.ndarray` of type `numpy.uint8` that describes the format
        representation of the given graph (resp. batch of graphs) in the adjacency matrix format
        with color numbers.
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__adjacency_matrix_colors is not None:
            return self.__adjacency_matrix_colors

        # Otherwise, perform the required conversion directly or indirectly by using the graph
        # format conversion graph.
        if (
            self.__adjacency_matrix_binary is not None
            or self.__bitmask_out is not None
            or self.__bitmask_in is not None
        ):
            self.adjacency_matrix_binary
            self.__convert_graph_format(
                input_format=GraphFormat.ADJACENCY_MATRIX_BINARY,
                output_format=GraphFormat.ADJACENCY_MATRIX_COLORS,
            )
        elif self.__flattened_row_major_colors is not None:
            self.__convert_graph_format(
                input_format=GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
                output_format=GraphFormat.ADJACENCY_MATRIX_COLORS,
            )
        elif self.__flattened_clockwise_colors is not None:
            self.__convert_graph_format(
                input_format=GraphFormat.FLATTENED_CLOCKWISE_COLORS,
                output_format=GraphFormat.ADJACENCY_MATRIX_COLORS,
            )
        elif self.__flattened_row_major_binary is not None:
            self.__convert_graph_format(
                input_format=GraphFormat.FLATTENED_ROW_MAJOR_BINARY,
                output_format=GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
            )
            self.__convert_graph_format(
                input_format=GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
                output_format=GraphFormat.ADJACENCY_MATRIX_COLORS,
            )
        else:
            self.__convert_graph_format(
                input_format=GraphFormat.FLATTENED_CLOCKWISE_BINARY,
                output_format=GraphFormat.FLATTENED_CLOCKWISE_COLORS,
            )
            self.__convert_graph_format(
                input_format=GraphFormat.FLATTENED_CLOCKWISE_COLORS,
                output_format=GraphFormat.ADJACENCY_MATRIX_COLORS,
            )

        return self.__adjacency_matrix_colors

    @property
    def adjacency_matrix_binary(self) -> np.ndarray:
        """
        This property returns the `numpy.ndarray` of type `numpy.uint8` that describes the format
        representation of the given graph (resp. batch of graphs) in the adjacency matrix format
        with binary slices.
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__adjacency_matrix_binary is not None:
            return self.__adjacency_matrix_binary

        # Otherwise, perform the required conversion directly or indirectly by using the graph
        # format conversion graph.
        if self.__bitmask_out is not None:
            self.__convert_graph_format(
                input_format=GraphFormat.BITMASK_OUT,
                output_format=GraphFormat.ADJACENCY_MATRIX_BINARY,
            )
        elif self.__bitmask_in is not None:
            self.__convert_graph_format(
                input_format=GraphFormat.BITMASK_IN,
                output_format=GraphFormat.ADJACENCY_MATRIX_BINARY,
            )
        elif self.__adjacency_matrix_colors is not None:
            self.__convert_graph_format(
                input_format=GraphFormat.ADJACENCY_MATRIX_COLORS,
                output_format=GraphFormat.ADJACENCY_MATRIX_BINARY,
            )
        elif self.__flattened_row_major_binary is not None:
            self.__convert_graph_format(
                input_format=GraphFormat.FLATTENED_ROW_MAJOR_BINARY,
                output_format=GraphFormat.ADJACENCY_MATRIX_BINARY,
            )
        elif self.__flattened_clockwise_binary is not None:
            self.__convert_graph_format(
                input_format=GraphFormat.FLATTENED_CLOCKWISE_BINARY,
                output_format=GraphFormat.ADJACENCY_MATRIX_BINARY,
            )
        else:
            if self.__flattened_row_major_colors is not None:
                self.__convert_graph_format(
                    input_format=GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
                    output_format=GraphFormat.ADJACENCY_MATRIX_COLORS,
                )
            else:
                self.__convert_graph_format(
                    input_format=GraphFormat.FLATTENED_CLOCKWISE_COLORS,
                    output_format=GraphFormat.ADJACENCY_MATRIX_COLORS,
                )

            self.__convert_graph_format(
                input_format=GraphFormat.ADJACENCY_MATRIX_COLORS,
                output_format=GraphFormat.ADJACENCY_MATRIX_BINARY,
            )

        return self.__adjacency_matrix_binary

    @property
    def flattened_row_major_colors(self) -> np.ndarray:
        """
        This property returns the `numpy.ndarray` of type `numpy.uint8` that describes the format
        representation of the given graph (resp. batch of graphs) in the flattened row-major format
        with color numbers.
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__flattened_row_major_colors is not None:
            return self.__flattened_row_major_colors

        # Otherwise, perform a direct conversion from the flattened row-major format with binary
        # slices, if possible.
        if self.__flattened_row_major_binary is not None:
            self.__convert_graph_format(
                input_format=GraphFormat.FLATTENED_ROW_MAJOR_BINARY,
                output_format=GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
            )
        # Otherwise, determine the output `np.ndarray` by relying on the adjacency matrix format
        # with color numbers.
        else:
            self.adjacency_matrix_colors
            self.__convert_graph_format(
                input_format=GraphFormat.ADJACENCY_MATRIX_COLORS,
                output_format=GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
            )

        return self.__flattened_row_major_colors

    @property
    def flattened_row_major_binary(self) -> np.ndarray:
        """
        This property returns the `numpy.ndarray` of type `numpy.uint8` that describes the format
        representation of the given graph (resp. batch of graphs) in the flattened row-major format
        with binary slices.
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__flattened_row_major_binary is not None:
            return self.__flattened_row_major_binary

        # Otherwise, perform the required conversion directly or indirectly by using the graph
        # format conversion graph.
        if self.__flattened_row_major_colors is not None:
            self.__convert_graph_format(
                input_format=GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
                output_format=GraphFormat.FLATTENED_ROW_MAJOR_BINARY,
            )
        elif (
            self.__adjacency_matrix_binary is not None
            or self.__bitmask_out is not None
            or self.__bitmask_in is not None
        ):
            self.adjacency_matrix_binary
            self.__convert_graph_format(
                input_format=GraphFormat.ADJACENCY_MATRIX_BINARY,
                output_format=GraphFormat.FLATTENED_ROW_MAJOR_BINARY,
            )
        elif self.__adjacency_matrix_colors is not None:
            self.__convert_graph_format(
                input_format=GraphFormat.ADJACENCY_MATRIX_COLORS,
                output_format=GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
            )
            self.__convert_graph_format(
                input_format=GraphFormat.FLATTENED_ROW_MAJOR_COLORS,
                output_format=GraphFormat.FLATTENED_ROW_MAJOR_BINARY,
            )
        else:
            self.flattened_clockwise_binary
            self.__convert_graph_format(
                input_format=GraphFormat.FLATTENED_CLOCKWISE_BINARY,
                output_format=GraphFormat.ADJACENCY_MATRIX_BINARY,
            )
            self.__convert_graph_format(
                input_format=GraphFormat.ADJACENCY_MATRIX_BINARY,
                output_format=GraphFormat.FLATTENED_ROW_MAJOR_BINARY,
            )

        return self.__flattened_row_major_binary

    @property
    def flattened_clockwise_colors(self) -> np.ndarray:
        """
        This property returns the `numpy.ndarray` of type `numpy.uint8` that describes the format
        representation of the given graph (resp. batch of graphs) in the flattened clockwise format
        with color numbers.
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__flattened_clockwise_colors is not None:
            return self.__flattened_clockwise_colors

        # Otherwise, perform a direct conversion from the flattened clockwise format with binary
        # slices, if possible.
        if self.__flattened_clockwise_binary is not None:
            self.__convert_graph_format(
                input_format=GraphFormat.FLATTENED_CLOCKWISE_BINARY,
                output_format=GraphFormat.FLATTENED_CLOCKWISE_COLORS,
            )
        # Otherwise, determine the output `np.ndarray` by relying on the adjacency matrix format
        # with color numbers.
        else:
            self.adjacency_matrix_colors
            self.__convert_graph_format(
                input_format=GraphFormat.ADJACENCY_MATRIX_COLORS,
                output_format=GraphFormat.FLATTENED_CLOCKWISE_COLORS,
            )

        return self.__flattened_clockwise_colors

    @property
    def flattened_clockwise_binary(self) -> np.ndarray:
        """
        This property returns the `numpy.ndarray` of type `numpy.uint8` that describes the format
        representation of the given graph (resp. batch of graphs) in the flattened clockwise format
        with binary slices.
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__flattened_clockwise_binary is not None:
            return self.__flattened_clockwise_binary

        # Otherwise, perform the required conversion directly or indirectly by using the graph
        # format conversion graph.
        if self.__flattened_clockwise_colors is not None:
            self.__convert_graph_format(
                input_format=GraphFormat.FLATTENED_CLOCKWISE_COLORS,
                output_format=GraphFormat.FLATTENED_CLOCKWISE_BINARY,
            )
        elif (
            self.__adjacency_matrix_binary is not None
            or self.__bitmask_out is not None
            or self.__bitmask_in is not None
        ):
            self.adjacency_matrix_binary
            self.__convert_graph_format(
                input_format=GraphFormat.ADJACENCY_MATRIX_BINARY,
                output_format=GraphFormat.FLATTENED_CLOCKWISE_BINARY,
            )
        elif self.__adjacency_matrix_colors is not None:
            self.__convert_graph_format(
                input_format=GraphFormat.ADJACENCY_MATRIX_COLORS,
                output_format=GraphFormat.FLATTENED_CLOCKWISE_COLORS,
            )
            self.__convert_graph_format(
                input_format=GraphFormat.FLATTENED_CLOCKWISE_COLORS,
                output_format=GraphFormat.FLATTENED_CLOCKWISE_BINARY,
            )
        else:
            self.flattened_row_major_binary
            self.__convert_graph_format(
                input_format=GraphFormat.FLATTENED_ROW_MAJOR_BINARY,
                output_format=GraphFormat.ADJACENCY_MATRIX_BINARY,
            )
            self.__convert_graph_format(
                input_format=GraphFormat.ADJACENCY_MATRIX_BINARY,
                output_format=GraphFormat.FLATTENED_CLOCKWISE_BINARY,
            )

        return self.__flattened_clockwise_binary

    def __getitem__(self, index: int) -> Graph:
        """
        If the given object models a batch of $k$-edge-colored looped complete graphs of the same
        order, then this method returns the graph in the batch with the provided index. On the
        other hand, if the given object models a $k$-edge-colored looped complete graph, then this
        method should not be used, and if it is, an `IndexError` is raised.

        :param index: The provided index that determines which $k$-edge-colored looped complete
            graph should be extracted from the batch. This argument must be a nonnegative `int`
            that is below the batch size of the given batch of graphs.

        :return: The graph in the batch with the provided index, given as a `Graph` object.
        """

        if self.__batch_size is None:
            raise IndexError

        # Extract all the computed graph format representations.
        bitmask_out = None if self.__bitmask_out is None else self.__bitmask_out[index]
        bitmask_in = None if self.__bitmask_in is None else self.__bitmask_in[index]
        adjacency_matrix_colors = (
            None
            if self.__adjacency_matrix_colors is None
            else self.__adjacency_matrix_colors[index]
        )
        adjacency_matrix_binary = (
            None
            if self.__adjacency_matrix_binary is None
            else self.__adjacency_matrix_binary[index]
        )
        flattened_row_major_colors = (
            None
            if self.__flattened_row_major_colors is None
            else self.__flattened_row_major_colors[index]
        )
        flattened_row_major_binary = (
            None
            if self.__flattened_row_major_binary is None
            else self.__flattened_row_major_binary[index]
        )
        flattened_clockwise_colors = (
            None
            if self.__flattened_clockwise_colors is None
            else self.__flattened_clockwise_colors[index]
        )
        flattened_clockwise_binary = (
            None
            if self.__flattened_clockwise_binary is None
            else self.__flattened_clockwise_binary[index]
        )

        return Graph(
            edge_colors=self.__edge_colors,
            is_directed=self.__is_directed,
            allow_loops=self.__allow_loops,
            bitmask_out=bitmask_out,
            bitmask_in=bitmask_in,
            adjacency_matrix_colors=adjacency_matrix_colors,
            adjacency_matrix_binary=adjacency_matrix_binary,
            flattened_row_major_colors=flattened_row_major_colors,
            flattened_row_major_binary=flattened_row_major_binary,
            flattened_clockwise_colors=flattened_clockwise_colors,
            flattened_clockwise_binary=flattened_clockwise_binary,
        )
