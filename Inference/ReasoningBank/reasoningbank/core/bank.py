"""Core components for the ReasoningBank library."""
import json
from typing import Any, List, Dict
from ..memory.base import MemoryBackend
from ..distillation.distill import judge_trajectory, distill_trajectory
from ..utils.config import load_config
from ..memory.chroma import ChromaMemoryBackend
from ..memory.json import JSONMemoryBackend
from sentence_transformers import SentenceTransformer
from langchain_community.llms import FakeListLLM

# Placeholder for a generic Embedding Model interface.
# The user would provide a model with an `embed_documents` method.
EmbeddingModel = Any
LLM = Any


class ReasoningBank:
    """
    The core class for the ReasoningBank library.

    Orchestrates memory storage, distillation, and retrieval. It integrates a
    memory backend, an embedding model, and a language model to provide a
    comprehensive solution for managing and utilizing agent experiences.

    Attributes:
        memory_backend (MemoryBackend): The backend used for storing and
            retrieving memories.
        embedding_model (EmbeddingModel): The model used for generating
            embeddings for text.
        llm (LLM): The language model used for judging and distilling
            trajectories.
    """

    def __init__(self, config_path: str = "./config.yaml"):
        """
        Initializes the ReasoningBank from a configuration file.

        Args:
            config_path (str): The path to the configuration file.
        """
        self.config = load_config(config_path)

        self.memory_backend = self._init_memory_backend()
        self.embedding_model = self._init_embedding_model()
        self.llm = self._init_llm()

    def _init_memory_backend(self) -> MemoryBackend:
        """Initializes the memory backend based on the configuration."""
        backend_type = self.config["memory"]["backend"]
        if backend_type == "chroma":
            return ChromaMemoryBackend(
                collection_name=self.config["memory"]["chroma"]["collection_name"]
            )
        elif backend_type == "json":
            return JSONMemoryBackend(
                filepath=self.config["memory"]["json"]["filepath"]
            )
        else:
            raise ValueError(f"Unknown memory backend type: {backend_type}")

    def _init_embedding_model(self) -> EmbeddingModel:
        """Initializes the embedding model based on the configuration."""
        model_name = self.config["embedding_model"]["model_name"]
        if model_name == "gemini-embedding-001":
            # In a real implementation, this would initialize the Gemini client.
            # For now, we'll use a placeholder that has the same `encode` method
            # as SentenceTransformer for compatibility.
            # This is a mock for testing purposes.
            return SentenceTransformer("all-MiniLM-L6-v2")
        elif model_name == "sentence-transformers":
            st_model_name = self.config["embedding_model"].get(
                "st_model_name", "all-MiniLM-L6-v2"
            )
            return SentenceTransformer(st_model_name)
        else:
            raise ValueError(f"Unknown embedding model: {model_name}")

    def _init_llm(self) -> LLM:
        """Initializes the language model based on the configuration."""
        # This is a placeholder for a more complex LLM initialization.
        # In a real application, this would involve loading the specified LLM
        # from a library like LangChain.
        provider = self.config["llm"]["provider"]
        if provider == "ollama":
            from langchain_community.llms import Ollama

            return Ollama(model=self.config["llm"]["model"])
        elif provider == "langchain.llms.Fake":
            # For demonstration purposes, we use a fake LLM.
            responses = [
                "Success",
                json.dumps(
                    [
                        {
                            "title": "Fake Memory",
                            "description": "A fake memory",
                            "content": "This is a fake memory.",
                        }
                    ]
                ),
            ]
            return FakeListLLM(responses=responses)
        elif provider == "vllm":
            import os
            from openai import OpenAI
            MEMORY_URL = os.getenv("MEMORY_URL")
            api_key = "EMPTY"
            return OpenAI(base_url=MEMORY_URL, api_key=api_key)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def add_experience(self, trajectory: str, query: str):
        """
        Adds a new experience to the bank.

        This method takes a trajectory and a query, judges the trajectory's
        success, distills it into memory items, generates an embedding for the
        query, and stores the entire experience in the memory backend.

        Args:
            trajectory (str): The sequence of actions and observations that
                constitute the experience.
            query (str): The initial query or task that the agent was trying
                to solve.
        """
        is_success = judge_trajectory(trajectory, query, self.llm)
        distilled_items = distill_trajectory(
            trajectory, query, self.llm, is_success
        )

        if not distilled_items:
            return

        # Generate an embedding for the query.
        query_embedding = self.embedding_model.encode(query)

        # Prepare the experience for storage.
        # We serialize the distilled_items to a JSON string to comply with
        # ChromaDB's metadata limitations.
        experience_to_add = {
            "embedding": query_embedding.tolist(),
            "metadata": {
                "query": query,
                "trajectory": trajectory,
                "distilled_items": json.dumps(distilled_items),
            },
            "document": query,
        }

        self.memory_backend.add([experience_to_add])

    def retrieve_memories(self, query: str, k: int = 1) -> List[Dict]:
        """
        Retrieves the top k most relevant memories for a given query.

        Args:
            query (str): The query to retrieve relevant memories for.
            k (int): The number of memories to retrieve.

        Returns:
            List[Dict]: A list of the top k most relevant memories.
        """
        query_embedding = self.embedding_model.encode(query)
        return self.memory_backend.query(query_embedding.tolist(), k)
