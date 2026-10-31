"""
Image processing for PyroChain.
"""

from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from .base import BaseProcessor


class ImageProcessor(BaseProcessor):
    """Image processor for feature extraction."""

    def __init__(
        self,
        device: torch.device,
        max_length: int = 512,
        image_size: tuple = (224, 224),
        model_name: str = "resnet50",
    ):
        """Initialize image processor."""
        super().__init__(device, max_length)
        self.image_size = image_size
        self.model_name = model_name

        # Setup image transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Load a simple feature extractor (in practice would use a trained model)
        self._setup_feature_extractor()

    def _setup_feature_extractor(self):
        """Setup the feature extraction model."""
        # In practice, this would load a pre-trained model
        # For now, we'll use a simple placeholder
        self.feature_dim = 2048  # ResNet50 feature dimension

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Process image data."""
        processed = data.copy()

        # Extract image fields
        image_fields = self._extract_image_fields(data)

        # Process each image field
        processed_images = {}
        for field, image in image_fields.items():
            if image is not None:
                processed_images[field] = self._process_image(image)

        processed["processed_images"] = processed_images
        processed["image_features"] = self._extract_image_features(processed_images)

        return processed

    def _extract_image_fields(self, data: dict[str, Any]) -> dict[str, Image.Image]:
        """Extract image fields from data."""
        image_fields = {}

        # Common image field names
        image_field_names = [
            "image",
            "img",
            "photo",
            "picture",
            "thumbnail",
            "preview",
            "banner",
            "logo",
            "icon",
            "avatar",
        ]

        for field in image_field_names:
            if field in data and data[field] is not None:
                try:
                    image = self._validate_image(data[field])
                    image_fields[field] = image
                except ValueError:
                    continue

        # Also check for other image-like fields (PIL images, or 3-D arrays/tensors such as
        # HxWxC / CxHxW). 1-D/2-D tensors are feature vectors, not images.
        for key, value in data.items():
            if key in image_fields:
                continue
            if isinstance(value, Image.Image) or (
                isinstance(value, (np.ndarray, torch.Tensor)) and value.ndim == 3
            ):
                try:
                    image = self._validate_image(value)
                    image_fields[key] = image
                except ValueError:
                    continue

        return image_fields

    def _process_image(self, image: Image.Image | np.ndarray | torch.Tensor) -> dict[str, Any]:
        """Process a single image."""
        # Validate and convert to PIL
        pil_image = self._validate_image(image)

        # Get basic image properties
        width, height = pil_image.size
        mode = pil_image.mode

        # Apply transforms (normalisation expects 3 channels)
        tensor_image = self.transform(pil_image.convert("RGB"))

        # Extract basic features
        features = self._extract_basic_features(pil_image)

        return {
            "image": pil_image,
            "tensor": tensor_image,
            "width": width,
            "height": height,
            "mode": mode,
            "aspect_ratio": width / height,
            "basic_features": features,
        }

    def _extract_basic_features(self, image: Image.Image) -> dict[str, Any]:
        """Extract basic features from image."""
        # Convert to numpy for analysis
        img_array = np.array(image)

        # Color analysis
        if len(img_array.shape) == 3:
            # RGB image
            r_mean = np.mean(img_array[:, :, 0])
            g_mean = np.mean(img_array[:, :, 1])
            b_mean = np.mean(img_array[:, :, 2])
            brightness = np.mean(img_array)
        else:
            # Grayscale image
            r_mean = g_mean = b_mean = np.mean(img_array)
            brightness = np.mean(img_array)

        # Edge detection
        gray = (
            cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        )
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # Texture analysis (simplified)
        texture_variance = np.var(gray)

        return {
            "r_mean": float(r_mean),
            "g_mean": float(g_mean),
            "b_mean": float(b_mean),
            "brightness": float(brightness),
            "edge_density": float(edge_density),
            "texture_variance": float(texture_variance),
        }

    def _extract_image_features(self, processed_images: dict[str, dict[str, Any]]) -> torch.Tensor:
        """Extract features from processed images."""
        if not processed_images:
            # Return zero tensor if no images
            return torch.zeros(1, self.feature_dim, device=self.device)

        # Combine features from all images
        all_features = []

        for image_data in processed_images.values():
            # Extract features (simplified - in practice would use a trained model)
            # For now, we'll use the basic features and create a feature vector
            basic_features = image_data["basic_features"]

            # Create feature vector from basic features
            feature_vector = torch.tensor(
                [
                    basic_features["r_mean"] / 255.0,
                    basic_features["g_mean"] / 255.0,
                    basic_features["b_mean"] / 255.0,
                    basic_features["brightness"] / 255.0,
                    basic_features["edge_density"],
                    basic_features["texture_variance"] / 10000.0,  # Normalize
                    image_data["aspect_ratio"],
                    image_data["width"] / 1000.0,  # Normalize
                    image_data["height"] / 1000.0,  # Normalize
                ],
                device=self.device,
            )

            # Pad or truncate to feature_dim
            if len(feature_vector) < self.feature_dim:
                padding = torch.zeros(self.feature_dim - len(feature_vector), device=self.device)
                feature_vector = torch.cat([feature_vector, padding])
            else:
                feature_vector = feature_vector[: self.feature_dim]

            all_features.append(feature_vector)

        # Average features across all images
        if all_features:
            combined_features = torch.stack(all_features).mean(dim=0).unsqueeze(0)
        else:
            combined_features = torch.zeros(1, self.feature_dim, device=self.device)

        return combined_features

    def get_features(self, data: dict[str, Any]) -> torch.Tensor:
        """Get image features from data."""
        processed = self.process(data)
        return processed["image_features"]

    def batch_process(self, data_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process a batch of data samples."""
        return [self.process(data) for data in data_list]

    def detect_objects(
        self, image: Image.Image | np.ndarray | torch.Tensor
    ) -> list[dict[str, Any]]:
        """Detect objects in image (simplified implementation)."""
        # This is a placeholder for object detection
        # In practice, would use a trained object detection model

        pil_image = self._validate_image(image)

        # Simple edge-based object detection
        img_array = np.array(pil_image)
        gray = (
            cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        )

        # Find contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small objects
                x, y, w, h = cv2.boundingRect(contour)
                objects.append(
                    {
                        "id": i,
                        "bbox": [x, y, w, h],
                        "area": float(area),
                        "confidence": min(1.0, area / 10000.0),  # Simple confidence score
                    }
                )

        return objects

    def extract_colors(
        self, image: Image.Image | np.ndarray | torch.Tensor, n_colors: int = 5
    ) -> list[dict[str, Any]]:
        """Extract dominant colors from image."""
        # RGB first: grayscale (2-D) and RGBA (4-channel) arrays broke the reshape below
        pil_image = self._validate_image(image).convert("RGB")

        # Resize image for faster processing
        # NEAREST keeps real pixel colours (smooth resampling invents blended edge colours)
        pil_image = pil_image.resize((150, 150), Image.Resampling.NEAREST)
        img_array = np.array(pil_image)

        # Reshape image to be a list of pixels
        pixels = img_array.reshape(-1, 3)

        # Simple k-means clustering for color extraction
        from sklearn.cluster import KMeans

        n_colors = max(1, min(n_colors, len(np.unique(pixels, axis=0))))
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        colors: list[dict[str, Any]] = []
        for i, color in enumerate(kmeans.cluster_centers_):
            r, g, b = (round(float(c)) for c in color)  # centers are floats; ":02x" needs ints
            colors.append(
                {
                    "rgb": [r, g, b],
                    "hex": f"#{r:02x}{g:02x}{b:02x}",
                    "percentage": float(np.sum(kmeans.labels_ == i) / len(pixels)),
                }
            )

        colors.sort(key=lambda c: c["percentage"], reverse=True)
        return colors
