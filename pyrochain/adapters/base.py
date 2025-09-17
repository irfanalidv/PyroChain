"""
Base adapter class for lightweight LLM fine-tuning.
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict
from transformers import AutoModel, AutoTokenizer


class BaseAdapter(ABC):
    """Abstract base class for lightweight adapters."""

    def __init__(self, model_name: str, device: torch.device, **kwargs):
        """Initialize the adapter."""
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the base model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @abstractmethod
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the adapter."""
        pass

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save adapter weights."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load adapter weights."""
        pass

    def encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Encode text using the tokenizer."""
        encoding = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        return {k: v.to(self.device) for k, v in encoding.items()}

    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get embeddings for text."""
        encoding = self.encode_text(text)
        with torch.no_grad():
            outputs = self.model(**encoding)
            # Use mean pooling of last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    def freeze_base_model(self):
        """Freeze the base model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self):
        """Unfreeze the base model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
