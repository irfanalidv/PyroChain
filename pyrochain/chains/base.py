"""
Base chain class for PyroChain chains.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage


class BaseChain(ABC):
    """Abstract base class for PyroChain chains."""

    def __init__(self, agent, memory_type: str = "conversation_buffer", **kwargs):
        """Initialize the chain."""
        self.agent = agent
        self.memory_type = memory_type
        self.memory = self._setup_memory()

    def _setup_memory(self):
        """Setup memory for the chain."""
        return self.agent.memory

    @abstractmethod
    def run(self, **kwargs) -> dict[str, Any]:
        """Run the chain."""
        pass

    def save_memory(self, path: str | Path):
        """Save chain memory."""
        self.agent.save_memory(path)

    def load_memory(self, path: str | Path):
        """Load chain memory."""
        self.agent.load_memory(path)

    def clear_memory(self):
        """Clear chain memory."""
        self.agent.clear_memory()

    def get_memory_summary(self) -> dict[str, Any]:
        """Get memory summary."""
        return self.agent.get_memory_summary()

    def add_to_memory(self, message: BaseMessage):
        """Add message to memory."""
        if hasattr(self.memory, "add_message"):
            self.memory.add_message(message)
        elif hasattr(self.memory, "chat_memory"):
            self.memory.chat_memory.add_message(message)

    def get_memory_messages(self) -> list[BaseMessage]:
        """Get messages from memory."""
        if hasattr(self.memory, "chat_memory"):
            return self.memory.chat_memory.messages
        return []

    def update_memory(self, new_messages: list[BaseMessage]):
        """Update memory with new messages."""
        for message in new_messages:
            self.add_to_memory(message)

    def get_chain_state(self) -> dict[str, Any]:
        """Get current chain state."""
        return {
            "memory_type": self.memory_type,
            "memory_summary": self.get_memory_summary(),
            "agent_type": type(self.agent).__name__,
        }

    def reset_chain(self):
        """Reset the chain state."""
        self.clear_memory()
        if hasattr(self.agent, "reset"):
            self.agent.reset()
