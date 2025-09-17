"""
Text processing for PyroChain.
"""

import torch
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer
from .base import BaseProcessor


class TextProcessor(BaseProcessor):
    """Text processor for feature extraction."""
    
    def __init__(
        self,
        device: torch.device,
        max_length: int = 512,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """Initialize text processor."""
        super().__init__(device, max_length)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
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
    
    def _extract_text_fields(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Extract text fields from data."""
        text_fields = {}
        
        # Common text field names
        text_field_names = [
            "text", "description", "title", "content", "summary",
            "name", "label", "category", "brand", "review"
        ]
        
        for field in text_field_names:
            if field in data and data[field]:
                text_fields[field] = self._normalize_text(data[field])
        
        # Also check for nested text fields
        for key, value in data.items():
            if isinstance(value, str) and value.strip():
                text_fields[key] = self._normalize_text(value)
        
        return text_fields
    
    def _process_text(self, text: str) -> Dict[str, Any]:
        """Process a single text string."""
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        return {
            "text": text,
            "tokens": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "length": len(text.split()),
            "char_length": len(text)
        }
    
    def _extract_text_features(self, processed_texts: Dict[str, Dict[str, Any]]) -> torch.Tensor:
        """Extract features from processed texts."""
        if not processed_texts:
            # Return zero tensor if no text
            return torch.zeros(1, 384, device=self.device)  # Default embedding size
        
        # Combine all texts
        combined_text = " ".join([text_data["text"] for text_data in processed_texts.values()])
        
        # Get embeddings (simplified - in practice would use a proper embedding model)
        encoding = self.tokenizer(
            combined_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Simple feature extraction (in practice would use a trained model)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Create simple embeddings (replace with actual embedding model)
        embeddings = torch.randn(1, 384, device=self.device)  # Placeholder
        
        return embeddings
    
    def get_features(self, data: Dict[str, Any]) -> torch.Tensor:
        """Get text features from data."""
        processed = self.process(data)
        return processed["text_features"]
    
    def batch_process(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of data samples."""
        return [self.process(data) for data in data_list]
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract keywords from text (simplified implementation)."""
        # Simple keyword extraction based on word frequency
        words = text.lower().split()
        
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "can", "this", "that", "these", "those"
        }
        
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count word frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top k
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]
    
    def extract_sentiment(self, text: str) -> Dict[str, float]:
        """Extract sentiment from text (simplified implementation)."""
        # Simple sentiment analysis based on word lists
        positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "love", "like", "best", "perfect", "awesome", "brilliant"
        }
        
        negative_words = {
            "bad", "terrible", "awful", "horrible", "worst", "hate", "dislike",
            "poor", "disappointing", "frustrating", "annoying", "useless"
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
            "neutral": max(0.0, neutral_score)
        }
