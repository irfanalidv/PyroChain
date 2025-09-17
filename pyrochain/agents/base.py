"""
Base agent class for PyroChain agents.
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate


class PyroChainLLM:
    """Custom LLM wrapper for PyroChain adapters."""

    def __init__(self, adapter, **kwargs):
        self.adapter = adapter

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate text using the adapter."""
        # Encode the prompt
        encoding = self.adapter.encode_text(prompt)

        # Generate response (simplified - in practice would use proper generation)
        with torch.no_grad():
            # This is a placeholder - in practice would use proper text generation
            response = f"Generated response for: {prompt[:50]}..."

        return response

    @property
    def _llm_type(self) -> str:
        return "pyrochain_adapter"


class BaseAgent(ABC):
    """Abstract base class for PyroChain agents."""

    def __init__(
        self,
        adapter,
        processor,
        memory_type: str = "conversation_buffer",
        max_memory_size: int = 1000,
        **kwargs,
    ):
        """Initialize the agent."""
        self.adapter = adapter
        self.processor = processor
        self.memory_type = memory_type
        self.max_memory_size = max_memory_size

        # Initialize memory
        self.memory = self._setup_memory()

        # Initialize LLM wrapper
        self.llm = PyroChainLLM(adapter)

        # Initialize tools
        self.tools = self._setup_tools()

        # Initialize agent
        self.agent = self._setup_agent()
        self.agent_executor = self._setup_agent_executor()

    def _setup_memory(self):
        """Setup memory for the agent."""
        if self.memory_type == "conversation_buffer":
            return ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                max_token_limit=self.max_memory_size,
            )
        elif self.memory_type == "conversation_summary":
            return ConversationSummaryMemory(
                llm=self.llm,
                memory_key="chat_history",
                return_messages=True,
                max_token_limit=self.max_memory_size,
            )
        else:
            return ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )

    @abstractmethod
    def _setup_tools(self) -> List[Any]:
        """Setup tools for the agent."""
        pass

    @abstractmethod
    def _setup_agent(self):
        """Setup the agent."""
        pass

    def _setup_agent_executor(self) -> AgentExecutor:
        """Setup the agent executor."""
        return AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
        )

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """Run the agent."""
        pass

    def save_memory(self, path: Union[str, Path]):
        """Save agent memory."""
        import pickle
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "memory.pkl", "wb") as f:
            pickle.dump(self.memory, f)

    def load_memory(self, path: Union[str, Path]):
        """Load agent memory."""
        import pickle
        from pathlib import Path

        path = Path(path)
        memory_file = path / "memory.pkl"

        if memory_file.exists():
            with open(memory_file, "rb") as f:
                self.memory = pickle.load(f)

    def clear_memory(self):
        """Clear agent memory."""
        self.memory.clear()

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of agent memory."""
        if hasattr(self.memory, "chat_memory"):
            messages = self.memory.chat_memory.messages
        else:
            messages = []

        return {
            "memory_type": self.memory_type,
            "num_messages": len(messages),
            "memory_size": self.max_memory_size,
        }
