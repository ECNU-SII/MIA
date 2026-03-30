"""ChromaDB memory backend."""

import chromadb
from typing import List, Dict
import uuid
from .base import MemoryBackend


class ChromaMemoryBackend(MemoryBackend):
    """
    A memory backend that uses ChromaDB for storage.

    This class provides an implementation of the MemoryBackend that uses
    ChromaDB to store and retrieve memories. It is suitable for production
    environments where a scalable and efficient vector database is required.
    """

    def __init__(self, collection_name: str = "reasoning_bank"):
        """
        Initializes the ChromaMemoryBackend.

        Args:
            collection_name (str): The name of the ChromaDB collection to use.
        """
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add(self, items: List[Dict]):
        """
        Adds a list of memory items to the ChromaDB collection.

        Each item is a dictionary that should contain:
        - embedding: The embedding of the memory item.
        - metadata: A dictionary with title, description, and content.
        - document: The content of the memory item.

        Args:
            items (List[Dict]): A list of memory items to add.
        """
        self.collection.add(
            ids=[str(uuid.uuid4()) for _ in items],
            embeddings=[item["embedding"] for item in items],
            metadatas=[item["metadata"] for item in items],
            documents=[item["document"] for item in items],
        )

    def query(self, query_embedding: List[float], k: int) -> List[Dict]:
        """
        Queries the ChromaDB collection for the k most similar items.

        Args:
            query_embedding (List[float]): The embedding of the query.
            k (int): The number of results to return.

        Returns:
            List[Dict]: A list of metadata dictionaries for the k most similar
            items.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=k
        )
        # The query returns a list of lists for metadatas, one for each query
        # embedding. Since we only pass one embedding, we take the first list
        # of results.
        return results["metadatas"][0] if results["metadatas"] else []
