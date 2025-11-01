from __future__ import annotations

from enum import Enum
from math import isqrt

import numpy as np


class EdgeOrdering(Enum):
    """
    #TODO
    """

    COLUMN_FIRST = 0
    ROW_FIRST = 1


class Graph:
    """
    #TODO
    """

    def __init__(self):
        raise TypeError(f"The Graph class cannot be instantiated directly.")

    @classmethod
    def from_bitmask_format(cls, bitmask_format: np.ndarray) -> Graph:
        """
        #TODO
        """

        self = object.__new__(cls)

        self.__edge_colors = bitmask_format.shape[0]
        self.__order = bitmask_format.shape[1]

        self.__bitmask_format = bitmask_format
        self.__adjacency_matrix = None
        self.__flattened_column_first = None
        self.__flattened_row_first = None

        return self

    @classmethod
    def from_adjacency_matrix(cls, adjacency_matrix: np.ndarray, edge_colors: int) -> Graph:
        """
        #TODO
        """

        self = object.__new__(cls)

        self.__edge_colors = edge_colors
        self.__order = adjacency_matrix.shape[0]

        self.__bitmask_format = None
        self.__adjacency_matrix = adjacency_matrix
        self.__flattened_column_first = None
        self.__flattened_row_first = None

        return self

    @classmethod
    def from_flattened_format(
        cls, flattened_format: np.ndarray, edge_ordering: EdgeOrdering, edge_colors: int
    ) -> Graph:
        """
        #TODO
        """

        self = object.__new__(cls)

        self.__edge_colors = edge_colors
        self.__order = (isqrt(8 * flattened_format.shape[0] + 1) + 1) // 2

        self.__bitmask_format = None
        self.__adjacency_matrix = None
        if edge_ordering == EdgeOrdering.COLUMN_FIRST:
            self.__flattened_column_first = flattened_format
            self.__flattened_row_first = None
        else:
            self.__flattened_column_first = None
            self.__flattened_row_first = flattened_format

        return self

    @property
    def edge_colors(self) -> int:
        """
        #TODO
        """

        return self.__edge_colors

    @property
    def order(self) -> int:
        """
        #TODO
        """

        return self.__order

    @property
    def bitmask_format(self) -> np.ndarray:
        """
        #TODO
        """

        if self.__bitmask_format is not None:
            return self.__bitmask_format

        color_indices = np.arange(self.__edge_colors, dtype=int)
        temp = (self.adjacency_matrix == color_indices[:, None, None]).astype(int)
        np.fill_diagonal(temp[0], 0)

        masks = 1 << np.arange(self.__order, dtype=int)
        result = temp @ masks

        self.__bitmask_format = result

        return self.__bitmask_format

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """
        #TODO
        """

        if self.__adjacency_matrix is not None:
            return self.__adjacency_matrix

        if self.__flattened_column_first is not None:
            tril_rows, tril_columns = np.tril_indices(self.__order, k=-1)

            result = np.zeros((self.__order, self.__order), dtype=int)
            result[tril_rows, tril_columns] = self.__flattened_column_first
            result[tril_columns, tril_rows] = self.__flattened_column_first

            self.__adjacency_matrix = result

            return self.__adjacency_matrix

        if self.__flattened_row_first is not None:
            triu_rows, triu_columns = np.triu_indices(self.__order, k=1)

            result = np.zeros((self.__order, self.__order), dtype=int)
            result[triu_rows, triu_columns] = self.__flattened_row_first
            result[triu_columns, triu_rows] = self.__flattened_row_first

            self.__adjacency_matrix = result

            return self.__adjacency_matrix

        masks = (1 << np.arange(self.__order, dtype=int)).reshape(-1, 1, 1)
        temp = (self.__bitmask_format & masks).transpose(1, 2, 0)
        temp = (temp != 0).astype(int)

        sums = np.cumsum(temp, axis=0)
        sums = np.sum(sums, axis=0)

        result = np.full((self.__order, self.__order), self.__edge_colors, dtype=int) - sums
        np.fill_diagonal(result, 0)

        self.__adjacency_matrix = result

        return self.__adjacency_matrix

    @property
    def flattened_column_first(self) -> np.ndarray:
        """
        #TODO
        """

        if self.__flattened_column_first is not None:
            return self.__flattened_column_first

        tril_indices = np.tril_indices(self.__order, k=-1)
        result = self.adjacency_matrix[tril_indices]

        self.__flattened_column_first = result

        return self.__flattened_column_first

    @property
    def flattened_row_first(self) -> np.ndarray:
        """
        #TODO
        """

        if self.__flattened_row_first is not None:
            return self.__flattened_row_first

        triu_indices = np.triu_indices(self.__order, k=1)
        result = self.adjacency_matrix[triu_indices]

        self.__flattened_row_first = result

        return self.__flattened_row_first


class GraphBatch:
    """
    #TODO
    """

    def __init__(self):
        raise TypeError(f"The GraphBatch class cannot be instantiated directly.")

    @classmethod
    def from_bitmask_format_batch(cls, bitmask_format_batch: np.ndarray) -> GraphBatch:
        """
        #TODO
        """

        self = object.__new__(cls)

        self.__batch_size = bitmask_format_batch.shape[0]
        self.__edge_colors = bitmask_format_batch.shape[1]
        self.__order = bitmask_format_batch.shape[2]

        self.__bitmask_format_batch = bitmask_format_batch
        self.__adjacency_matrix_batch = None
        self.__flattened_column_first_batch = None
        self.__flattened_row_first_batch = None

        return self

    @classmethod
    def from_adjacency_matrix_batch(
        cls, adjacency_matrix_batch: np.ndarray, edge_colors: int
    ) -> GraphBatch:
        """
        #TODO
        """

        self = object.__new__(cls)

        self.__batch_size = adjacency_matrix_batch.shape[0]
        self.__edge_colors = edge_colors
        self.__order = adjacency_matrix_batch.shape[1]

        self.__bitmask_format_batch = None
        self.__adjacency_matrix_batch = adjacency_matrix_batch
        self.__flattened_column_first_batch = None
        self.__flattened_row_first_batch = None

        return self

    @classmethod
    def from_flattened_format_batch(
        cls, flattened_format_batch: np.ndarray, edge_ordering: EdgeOrdering, edge_colors: int
    ) -> GraphBatch:
        """
        #TODO
        """

        self = object.__new__(cls)

        self.__batch_size = flattened_format_batch.shape[0]
        self.__edge_colors = edge_colors
        self.__order = (isqrt(8 * flattened_format_batch.shape[1] + 1) + 1) // 2

        self.__bitmask_format_batch = None
        self.__adjacency_matrix_batch = None
        if edge_ordering == EdgeOrdering.COLUMN_FIRST:
            self.__flattened_column_first_batch = flattened_format_batch
            self.__flattened_row_first_batch = None
        else:
            self.__flattened_column_first_batch = None
            self.__flattened_row_first_batch = flattened_format_batch

        return self

    @property
    def batch_size(self) -> int:
        """
        #TODO
        """

        return self.__batch_size

    @property
    def edge_colors(self) -> int:
        """
        #TODO
        """

        return self.__edge_colors

    @property
    def order(self) -> int:
        """
        #TODO
        """

        return self.__order

    @property
    def bitmask_format_batch(self) -> np.ndarray:
        """
        #TODO
        """

        if self.__bitmask_format_batch is not None:
            return self.__bitmask_format_batch

        color_indices = np.arange(self.__edge_colors, dtype=int)
        temp = (
            self.adjacency_matrix_batch[:, None, :, :] == color_indices[None, :, None, None]
        ).astype(int)
        np.einsum("bcii->bci", temp)[:] = 0

        masks = 1 << np.arange(self.__order, dtype=int)
        result = temp @ masks

        self.__bitmask_format_batch = result

        return self.__bitmask_format_batch

    @property
    def adjacency_matrix_batch(self) -> np.ndarray:
        """
        #TODO
        """

        if self.__adjacency_matrix_batch is not None:
            return self.__adjacency_matrix_batch

        if self.__flattened_column_first_batch is not None:
            tril_rows, tril_columns = np.tril_indices(self.__order, k=-1)

            result = np.zeros((self.__batch_size, self.__order, self.__order), dtype=int)
            result[:, tril_rows, tril_columns] = self.__flattened_column_first_batch
            result[:, tril_columns, tril_rows] = self.__flattened_column_first_batch

            self.__adjacency_matrix_batch = result

            return self.__adjacency_matrix_batch

        if self.__flattened_row_first_batch is not None:
            triu_rows, triu_columns = np.triu_indices(self.__order, k=1)

            result = np.zeros((self.__batch_size, self.__order, self.__order), dtype=int)
            result[:, triu_rows, triu_columns] = self.__flattened_row_first_batch
            result[:, triu_columns, triu_rows] = self.__flattened_row_first_batch

            self.__adjacency_matrix_batch = result

            return self.__adjacency_matrix_batch

        masks = (1 << np.arange(self.__order, dtype=int)).reshape(-1, 1, 1, 1)
        temp = (self.__bitmask_format_batch & masks).transpose(1, 2, 3, 0)
        temp = (temp != 0).astype(int)

        sums = np.cumsum(temp, axis=1)
        sums = np.sum(sums, axis=1)

        result = (
            np.full((self.__batch_size, self.__order, self.__order), self.__edge_colors, dtype=int)
            - sums
        )
        np.einsum("bii->bi", result)[:] = 0

        self.__adjacency_matrix_batch = result

        return self.__adjacency_matrix_batch

    @property
    def flattened_column_first_batch(self) -> np.ndarray:
        """
        #TODO
        """

        if self.__flattened_column_first_batch is not None:
            return self.__flattened_column_first_batch

        tril_rows, tril_columns = np.tril_indices(self.__order, k=-1)
        result = self.adjacency_matrix_batch[:, tril_rows, tril_columns]

        self.__flattened_column_first_batch = result

        return self.__flattened_column_first_batch

    @property
    def flattened_row_first_batch(self) -> np.ndarray:
        """
        #TODO
        """

        if self.__flattened_row_first_batch is not None:
            return self.__flattened_row_first_batch

        triu_rows, triu_columns = np.triu_indices(self.__order, k=1)
        result = self.adjacency_matrix_batch[:, triu_rows, triu_columns]

        self.__flattened_row_first_batch = result

        return self.__flattened_row_first_batch
