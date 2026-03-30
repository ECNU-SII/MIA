"""JSON file memory backend."""

import json
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .base import MemoryBackend


class JSONMemoryBackend(MemoryBackend):
    """
    A simple memory backend that stores memories in a JSON file.

    This implementation is intended for testing and development purposes. It
    stores memories in a simple JSON file and uses cosine similarity to perform
    queries. It is not recommended for production use due to performance
    limitations.
    """

    def __init__(self, filepath: str):
        """
        Initializes the JSONMemoryBackend.

        Args:
            filepath (str): The path to the JSON file where memories will be
                stored.
        """
        self.filepath = filepath
        self.data = self._load()

    def _load(self) -> List[Dict]:
        """Loads memories from the JSON file."""
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def _save(self):
        """Saves memories to the JSON file."""
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=4)

    def add(self, items: List[Dict]):
        """
        Adds a list of memory items to the JSON file.

        Args:
            items (List[Dict]): A list of memory items to add.
        """
        self.data.extend(items)
        self._save()

    def query(self, query_embedding: List[float], k: int) -> List[Dict]:
        """
        Queries the JSON file for the k most similar items using cosine
        similarity.

        Args:
            query_embedding (List[float]): The embedding of the query.
            k (int): The number of results to return.

        Returns:
            List[Dict]: A list of metadata dictionaries for the k most similar
            items.
        """
        if not self.data:
            return []

        embeddings = np.array([item["embedding"] for item in self.data])
        query_embedding_np = np.array(query_embedding).reshape(1, -1)

        similarities = cosine_similarity(query_embedding_np, embeddings)[0]

        # Get the indices of the top k most similar items
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        return [self.data[i]["metadata"] for i in top_k_indices]
