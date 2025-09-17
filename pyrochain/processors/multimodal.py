"""
Multimodal processor that combines text and image processing.
"""

import torch
from typing import Dict, Any, List, Optional, Union
from .base import BaseProcessor
from .text import TextProcessor
from .image import ImageProcessor


class MultimodalProcessor(BaseProcessor):
    """
    Multimodal processor that handles both text and image data.
    
    Combines text and image processing capabilities for comprehensive
    feature extraction from multimodal datasets.
    """
    
    def __init__(
        self,
        adapter,
        device: torch.device,
        max_length: int = 512,
        image_size: tuple = (224, 224),
        fusion_method: str = "concat"
    ):
        """Initialize multimodal processor."""
        super().__init__(device, max_length)
        self.adapter = adapter
        self.fusion_method = fusion_method
        
        # Initialize sub-processors
        self.text_processor = TextProcessor(
            device=device,
            max_length=max_length
        )
        
        self.image_processor = ImageProcessor(
            device=device,
            max_length=max_length,
            image_size=image_size
        )
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process multimodal data."""
        processed = data.copy()
        
        # Process text data
        text_processed = self.text_processor.process(data)
        processed.update(text_processed)
        
        # Process image data
        image_processed = self.image_processor.process(data)
        processed.update(image_processed)
        
        # Fuse multimodal features
        processed["multimodal_features"] = self._fuse_features(
            text_processed.get("text_features"),
            image_processed.get("image_features")
        )
        
        # Add metadata
        processed["modalities"] = self._detect_modalities(data)
        processed["processing_metadata"] = {
            "text_fields": list(text_processed.get("processed_texts", {}).keys()),
            "image_fields": list(image_processed.get("processed_images", {}).keys()),
            "fusion_method": self.fusion_method
        }
        
        return processed
    
    def _fuse_features(
        self, 
        text_features: Optional[torch.Tensor], 
        image_features: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Fuse text and image features."""
        if text_features is None and image_features is None:
            # Return zero tensor if no features
            return torch.zeros(1, 512, device=self.device)
        
        if text_features is None:
            return image_features
        
        if image_features is None:
            return text_features
        
        # Fuse features based on method
        if self.fusion_method == "concat":
            # Concatenate features
            return torch.cat([text_features, image_features], dim=-1)
        
        elif self.fusion_method == "add":
            # Add features (requires same dimension)
            if text_features.shape[-1] == image_features.shape[-1]:
                return text_features + image_features
            else:
                # Pad or truncate to match dimensions
                max_dim = max(text_features.shape[-1], image_features.shape[-1])
                text_padded = self._pad_or_truncate(text_features, max_dim)
                image_padded = self._pad_or_truncate(image_features, max_dim)
                return text_padded + image_padded
        
        elif self.fusion_method == "multiply":
            # Element-wise multiplication (requires same dimension)
            if text_features.shape[-1] == image_features.shape[-1]:
                return text_features * image_features
            else:
                # Pad or truncate to match dimensions
                max_dim = max(text_features.shape[-1], image_features.shape[-1])
                text_padded = self._pad_or_truncate(text_features, max_dim)
                image_padded = self._pad_or_truncate(image_features, max_dim)
                return text_padded * image_padded
        
        elif self.fusion_method == "attention":
            # Simple attention-based fusion
            return self._attention_fusion(text_features, image_features)
        
        else:
            # Default to concatenation
            return torch.cat([text_features, image_features], dim=-1)
    
    def _pad_or_truncate(self, tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Pad or truncate tensor to target dimension."""
        current_dim = tensor.shape[-1]
        
        if current_dim < target_dim:
            # Pad with zeros
            padding_size = target_dim - current_dim
            padding = torch.zeros(tensor.shape[:-1] + (padding_size,), device=tensor.device)
            return torch.cat([tensor, padding], dim=-1)
        elif current_dim > target_dim:
            # Truncate
            return tensor[..., :target_dim]
        else:
            return tensor
    
    def _attention_fusion(self, text_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        """Attention-based fusion of text and image features."""
        # Simple attention mechanism
        # In practice, this would be more sophisticated
        
        # Ensure same dimensions
        max_dim = max(text_features.shape[-1], image_features.shape[-1])
        text_features = self._pad_or_truncate(text_features, max_dim)
        image_features = self._pad_or_truncate(image_features, max_dim)
        
        # Compute attention weights
        text_norm = torch.norm(text_features, dim=-1, keepdim=True)
        image_norm = torch.norm(image_features, dim=-1, keepdim=True)
        
        # Simple attention weights based on feature magnitude
        text_weight = text_norm / (text_norm + image_norm + 1e-8)
        image_weight = image_norm / (text_norm + image_norm + 1e-8)
        
        # Weighted combination
        fused_features = text_weight * text_features + image_weight * image_features
        
        return fused_features
    
    def _detect_modalities(self, data: Dict[str, Any]) -> List[str]:
        """Detect available modalities in the data."""
        modalities = []
        
        # Check for text data
        text_fields = self.text_processor._extract_text_fields(data)
        if text_fields:
            modalities.append("text")
        
        # Check for image data
        image_fields = self.image_processor._extract_image_fields(data)
        if image_fields:
            modalities.append("image")
        
        return modalities
    
    def get_features(self, data: Dict[str, Any]) -> torch.Tensor:
        """Get multimodal features from data."""
        processed = self.process(data)
        return processed["multimodal_features"]
    
    def batch_process(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of multimodal data samples."""
        return [self.process(data) for data in data_list]
    
    def extract_structured_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured features from multimodal data."""
        processed = self.process(data)
        
        structured_features = {
            "text_features": {},
            "image_features": {},
            "multimodal_features": {},
            "metadata": processed["processing_metadata"]
        }
        
        # Extract text features
        if "processed_texts" in processed:
            for field, text_data in processed["processed_texts"].items():
                structured_features["text_features"][field] = {
                    "length": text_data["length"],
                    "char_length": text_data["char_length"],
                    "keywords": self.text_processor.extract_keywords(text_data["text"]),
                    "sentiment": self.text_processor.extract_sentiment(text_data["text"])
                }
        
        # Extract image features
        if "processed_images" in processed:
            for field, image_data in processed["processed_images"].items():
                structured_features["image_features"][field] = {
                    "dimensions": (image_data["width"], image_data["height"]),
                    "aspect_ratio": image_data["aspect_ratio"],
                    "basic_features": image_data["basic_features"],
                    "objects": self.image_processor.detect_objects(image_data["image"]),
                    "colors": self.image_processor.extract_colors(image_data["image"])
                }
        
        # Add multimodal fusion info
        structured_features["multimodal_features"] = {
            "fusion_method": self.fusion_method,
            "feature_dimension": processed["multimodal_features"].shape[-1],
            "modalities": processed["modalities"]
        }
        
        return structured_features
    
    def get_feature_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of extracted features."""
        structured_features = self.extract_structured_features(data)
        
        summary = {
            "modalities": structured_features["metadata"]["modalities"],
            "text_summary": {
                "num_text_fields": len(structured_features["text_features"]),
                "total_text_length": sum(
                    text_data["length"] for text_data in structured_features["text_features"].values()
                )
            },
            "image_summary": {
                "num_image_fields": len(structured_features["image_features"]),
                "total_objects": sum(
                    len(image_data["objects"]) for image_data in structured_features["image_features"].values()
                )
            },
            "multimodal_summary": {
                "fusion_method": structured_features["multimodal_features"]["fusion_method"],
                "feature_dimension": structured_features["multimodal_features"]["feature_dimension"]
            }
        }
        
        return summary
