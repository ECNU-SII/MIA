"""Abstract base classes and implementations for memory backends."""

import abc
from typing import List, Dict


class MemoryBackend(abc.ABC):
    """
    Abstract base class for memory backends.

    This class defines the interface that all memory backends must implement.
    It provides a standardized way to add and query memories, ensuring that
    different storage solutions can be used interchangeably.
    """

    @abc.abstractmethod
    def add(self, items: List[Dict]):
        """
        Adds a list of memory items to the backend.

        Args:
            items (List[Dict]): A list of dictionaries, where each dictionary
                represents a memory item to be added.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def query(self, query_embedding: List[float], k: int) -> List[Dict]:
        """
        Queries the backend for the k most similar items.

        Args:
            query_embedding (List[float]): The embedding of the query.
            k (int): The number of most similar items to retrieve.

        Returns:
            List[Dict]: A list of the top k most similar memory items.
        """
        raise NotImplementedError
