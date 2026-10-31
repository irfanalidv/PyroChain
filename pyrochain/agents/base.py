"""
Base agent class for PyroChain agents.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class ConversationMemory:
    """Bounded conversation memory holding LangChain ``BaseMessage`` objects.

    Replaces ``langchain.memory.ConversationBufferMemory``, which was removed in LangChain 1.x.
    ``chat_memory`` returns the memory itself so existing ``memory.chat_memory.messages`` /
    ``memory.chat_memory.add_message(...)`` code keeps working.
    """

    def __init__(self, max_messages: int | None = None, memory_key: str = "chat_history"):
        self.memory_key = memory_key
        self.max_messages = max_messages
        self.messages: list[BaseMessage] = []

    @property
    def chat_memory(self) -> "ConversationMemory":
        return self

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        if self.max_messages and len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def clear(self) -> None:
        self.messages = []


class SequentialToolExecutor:
    """Runs each of an agent's tools once, in order, and returns a ReAct-style transcript.

    PyroChain's agents do their real work in the multimodal processor; their tools produce a
    reasoning trace. This deterministic executor replaces LangChain's removed
    ``initialize_agent``/``AgentExecutor`` (which also required a LangChain chat model).
    """

    def __init__(self, tools: list[Any], memory: ConversationMemory | None = None):
        self.tools = tools
        self.memory = memory

    def invoke(self, input: str) -> str:  # noqa: A002 - mirrors LangChain's keyword
        lines = [f"Question: {input.strip()}"]
        for tool in self.tools:
            observation = tool.invoke(input)
            lines += [
                f"Thought: use {tool.name}",
                f"Action: {tool.name}",
                f"Observation: {observation}",
            ]
        lines.append("Final Answer: " + "; ".join(str(t.name) for t in self.tools) + " completed")
        transcript = "\n".join(lines)
        if self.memory is not None:
            self.memory.add_message(HumanMessage(content=input))
            self.memory.add_message(AIMessage(content=transcript))
        return transcript

    def run(self, input: str) -> str:  # noqa: A002 - backwards-compatible alias
        return self.invoke(input)


class PyroChainLLM:
    """Custom LLM wrapper for PyroChain adapters."""

    def __init__(self, adapter, **kwargs):
        self.adapter = adapter

    def __call__(self, prompt: str, stop: list[str] | None = None) -> str:
        """Generate text using the adapter."""
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

    def _setup_memory(self) -> ConversationMemory:
        """Setup memory for the agent.

        ``conversation_buffer`` and ``conversation_summary`` both map to a bounded buffer
        (summaries would need a real LLM); ``max_memory_size`` caps the number of messages.
        """
        return ConversationMemory(max_messages=self.max_memory_size, memory_key="chat_history")

    @abstractmethod
    def _setup_tools(self) -> list[Any]:
        """Setup tools for the agent."""
        pass

    @abstractmethod
    def _setup_agent(self):
        """Setup the agent."""
        pass

    def _setup_agent_executor(self) -> SequentialToolExecutor:
        """Setup the agent executor."""
        return SequentialToolExecutor(tools=self.tools, memory=self.memory)

    @abstractmethod
    def run(self, **kwargs) -> dict[str, Any]:
        """Run the agent."""
        pass

    def save_memory(self, path: str | Path) -> None:
        """Save agent memory as JSON (``memory.json``)."""
        import json

        from langchain_core.messages import messages_to_dict

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "memory.json", "w", encoding="utf-8") as f:
            json.dump(messages_to_dict(self.memory.messages), f)

    def load_memory(self, path: str | Path) -> None:
        """Load agent memory saved by :meth:`save_memory`."""
        import json

        from langchain_core.messages import messages_from_dict

        memory_file = Path(path) / "memory.json"
        if memory_file.exists():
            with open(memory_file, encoding="utf-8") as f:
                messages = messages_from_dict(json.load(f))
            self.memory.clear()
            for message in messages:
                self.memory.add_message(message)

    def clear_memory(self):
        """Clear agent memory."""
        self.memory.clear()

    def get_memory_summary(self) -> dict[str, Any]:
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
