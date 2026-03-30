from typing import Dict, Any, List
from reasoningbank.core.bank import ReasoningBank


class ReasoningBankMemory:
    """A LangChain memory class that uses the ReasoningBank."""

    reasoning_bank: ReasoningBank
    memory_key: str = "history"  # The key for the memory variables.

    def __init__(self, reasoning_bank: ReasoningBank):
        self.reasoning_bank = reasoning_bank

    @property
    def memory_variables(self) -> List[str]:
        """The list of memory variables."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load the memory variables."""
        import json

        # We'll use the first key in the inputs dict as the query.
        # This is a simplification; a more robust implementation might
        # have a more explicit way of defining the query.
        query = next(iter(inputs.values()))

        retrieved_experiences = self.reasoning_bank.retrieve_memories(query, k=1)

        all_distilled_items = []
        for experience in retrieved_experiences:
            # distilled_items is a JSON string, so we need to parse it.
            distilled_items = json.loads(experience.get("distilled_items", "[]"))
            all_distilled_items.extend(distilled_items)

        # Format the memories into a string for the prompt.
        formatted_memories = "\n---\n".join(
            f"Title: {item['title']}\nDescription: {item['description']}\nContent: {item['content']}"
            for item in all_distilled_items
        )

        return {self.memory_key: formatted_memories}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context of a chain run to the ReasoningBank."""
        query = next(iter(inputs.values()))
        trajectory = next(iter(outputs.values()))

        self.reasoning_bank.add_experience(trajectory, query)

    def clear(self) -> None:
        """Clear the memory."""
        # This is not directly supported by the current ReasoningBank design,
        # as the bank is meant to be persistent.
        # A possible implementation would be to clear the memory backend.
        pass
