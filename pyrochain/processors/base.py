"""
Base processor class for multimodal data processing.
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from PIL import Image
import numpy as np


class BaseProcessor(ABC):
    """Abstract base class for data processors."""
    
    def __init__(self, device: torch.device, max_length: int = 512):
        """Initialize the processor."""
        self.device = device
        self.max_length = max_length
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data."""
        pass
    
    @abstractmethod
    def get_features(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract features from processed data."""
        pass
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text input."""
        if not isinstance(text, str):
            text = str(text)
        return text.strip()
    
    def _validate_image(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """Validate and convert image to PIL Image."""
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL
            if image.dim() == 4:
                image = image.squeeze(0)
            if image.dim() == 3 and image.shape[0] in [1, 3]:
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
        
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        if not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        return image
    
    def _resize_image(self, image: Image.Image, size: tuple = (224, 224)) -> Image.Image:
        """Resize image to specified size."""
        return image.resize(size, Image.Resampling.LANCZOS)
    
    def _normalize_image(self, image: Image.Image) -> torch.Tensor:
        """Normalize image to tensor."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to tensor
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Normalize using ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        # Convert to tensor and rearrange dimensions
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        
        return image_tensor.to(self.device)
