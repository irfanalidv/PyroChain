"""
Text processing for PyroChain.
"""

from collections.abc import Callable
from typing import Any

import torch
from transformers import AutoTokenizer

from .base import BaseProcessor


class TextProcessor(BaseProcessor):
    """Text processor for feature extraction."""

    def __init__(
        self,
        device: torch.device,
        max_length: int = 512,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedder: Callable[[str], torch.Tensor] | None = None,
        feature_dim: int = 384,
    ):
        """Initialize text processor.

        Args:
            embedder: Optional callable mapping text to a ``(1, dim)`` embedding tensor (e.g.
                ``LightweightAdapter.get_embeddings``). Without it, a deterministic hashed
                bag-of-tokens vector of size ``feature_dim`` is used.
        """
        super().__init__(device, max_length)
        self.model_name = model_name
        self.embedder = embedder
        self.feature_dim = feature_dim
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Process text data."""
        processed = data.copy()

        # Extract text fields
        text_fields = self._extract_text_fields(data)

        # Process each text field
        processed_texts = {}
        for field, text in text_fields.items():
            if text:
                processed_texts[field] = self._process_text(text)

        processed["processed_texts"] = processed_texts
        processed["text_features"] = self._extract_text_features(processed_texts)

        return processed

    def _extract_text_fields(self, data: dict[str, Any]) -> dict[str, str]:
        """Extract text fields from data."""
        text_fields = {}

        # Common text field names
        text_field_names = [
            "text",
            "description",
            "title",
            "content",
            "summary",
            "name",
            "label",
            "category",
            "brand",
            "review",
        ]

        for field in text_field_names:
            if data.get(field):
                text_fields[field] = self._normalize_text(data[field])

        # Also check for nested text fields
        for key, value in data.items():
            if isinstance(value, str) and value.strip():
                text_fields[key] = self._normalize_text(value)

        return text_fields

    def _process_text(self, text: str) -> dict[str, Any]:
        """Process a single text string."""
        # Tokenize
        encoding = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
        )

        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        return {
            "text": text,
            "tokens": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "length": len(text.split()),
            "char_length": len(text),
        }

    def _extract_text_features(self, processed_texts: dict[str, dict[str, Any]]) -> torch.Tensor:
        """Extract features from processed texts."""
        if not processed_texts:
            # Return zero tensor if no text
            return torch.zeros(1, self.feature_dim, device=self.device)

        # Combine all texts
        combined_text = " ".join([text_data["text"] for text_data in processed_texts.values()])

        if self.embedder is not None:
            with torch.no_grad():
                return self.embedder(combined_text).to(self.device)

        # Deterministic fallback: L2-normalised hashed bag of token ids
        encoding = self.tokenizer(
            combined_text, truncation=True, max_length=self.max_length, add_special_tokens=False
        )
        ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        features = torch.zeros(self.feature_dim)
        if len(ids):
            features.index_add_(0, ids % self.feature_dim, torch.ones(len(ids)))
            features = features / features.norm()
        return features.unsqueeze(0).to(self.device)

    def get_features(self, data: dict[str, Any]) -> torch.Tensor:
        """Get text features from data."""
        processed = self.process(data)
        return processed["text_features"]

    def batch_process(self, data_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process a batch of data samples."""
        return [self.process(data) for data in data_list]

    def extract_keywords(self, text: str, top_k: int = 10) -> list[str]:
        """Extract keywords from text (simplified implementation)."""
        # Simple keyword extraction based on word frequency
        words = text.lower().split()

        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
        }

        words = [word for word in words if word not in stop_words and len(word) > 2]

        # Count word frequency
        word_freq: dict[str, int] = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and return top k
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]

    def extract_sentiment(self, text: str) -> dict[str, float]:
        """Extract sentiment from text (simplified implementation)."""
        # Simple sentiment analysis based on word lists
        positive_words = {
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
            "love",
            "like",
            "best",
            "perfect",
            "awesome",
            "brilliant",
        }

        negative_words = {
            "bad",
            "terrible",
            "awful",
            "horrible",
            "worst",
            "hate",
            "dislike",
            "poor",
            "disappointing",
            "frustrating",
            "annoying",
            "useless",
        }

        words = text.lower().split()

        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        total_words = len(words)

        if total_words == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        neutral_score = 1.0 - positive_score - negative_score

        return {
            "positive": positive_score,
            "negative": negative_score,
            "neutral": max(0.0, neutral_score),
        }
