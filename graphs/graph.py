from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np


class Graph:
    """
    #TODO
    """

    def __init__(self, graph: np.ndarray):
        self.__graph: np.ndarray = graph

    @property
    def data(self) -> np.ndarray:
        """
        #TODO
        """

        return self.__graph


class GraphBatch:
    """
    #TODO
    """

    def __init__(self):
        raise TypeError


    @classmethod
    def from_bitmask_format(cls, bitmask_format: np.ndarray) -> GraphBatch:
        self = object.__new__(cls)

        self.__bitmask_format = bitmask_format
        self.__adjacency_matrix = None
        self.__flat_column_first = None
        self.__flat_row_first = None

        self.__batch_size = bitmask_format.shape[0]
        self.__edge_colors = bitmask_format.shape[1]
        self.__order = bitmask_format.shape[2]

        return self

    @property
    def bitmask_format(self) -> np.ndarray:
        """
        #TODO
        """

        if self.__bitmask_format is not None:
            return self.__bitmask_format
        
        raise NotImplementedError

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """
        #TODO
        """

        if self.__adjacency_matrix is not None:
            return self.__adjacency_matrix
        
        if self.__bitmask_format is not None:
            masks = (1 << np.arange(self.__order, dtype=int)).reshape(-1, 1, 1, 1)
            temp = (self.__bitmask_format & masks).transpose(1, 2, 3, 0)
            temp = (temp != 0).astype(int)
            temp = np.flip(temp, axis=1)

            sums = 

            result = np.zeros((self.__batch_size, self.__order, self.__order), dtype=int)


            for color in range(self.__edge_colors):
                result += temp[:, color, :, :] * color
            
            self.__adjacency_matrix = result
            
            return self.__adjacency_matrix


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



class EdgeOrdering(Enum):
    """
    #TODO
    """

    COLUMN_FIRST = 0
    ROW_FIRST = 1
