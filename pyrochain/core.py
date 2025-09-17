"""
Core PyroChain functionality for orchestrating feature engineering pipelines.
"""

import torch
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from .adapters import LightweightAdapter
from .agents import FeatureAgent, ValidationAgent
from .chains import FeatureExtractionChain, ValidationChain
from .processors import MultimodalProcessor


@dataclass
class PyroChainConfig:
    """Configuration for PyroChain pipeline."""

    model_name: str = "microsoft/DialoGPT-small"
    adapter_rank: int = 16
    max_length: int = 512
    device: str = "auto"
    memory_type: str = "conversation_buffer"
    enable_validation: bool = True
    validation_threshold: float = 0.8


class PyroChain:
    """
    Main PyroChain class that orchestrates the feature engineering pipeline.

    Combines PyTorch adapters, LangChain agents, and multimodal processing
    for automated feature extraction with agentic validation.
    """

    def __init__(self, config: Optional[PyroChainConfig] = None):
        """Initialize PyroChain with configuration."""
        self.config = config or PyroChainConfig()
        self.device = self._setup_device()

        # Initialize components
        self.adapter = None
        self.processor = None
        self.feature_agent = None
        self.validation_agent = None
        self.feature_chain = None
        self.validation_chain = None

        self._initialize_components()

    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)

    def _initialize_components(self):
        """Initialize all PyroChain components."""
        # Initialize lightweight adapter
        self.adapter = LightweightAdapter(
            model_name=self.config.model_name,
            rank=self.config.adapter_rank,
            device=self.device,
        )

        # Initialize multimodal processor
        self.processor = MultimodalProcessor(
            adapter=self.adapter, device=self.device, max_length=self.config.max_length
        )

        # Initialize agents
        self.feature_agent = FeatureAgent(
            adapter=self.adapter, processor=self.processor
        )

        if self.config.enable_validation:
            self.validation_agent = ValidationAgent(
                adapter=self.adapter,
                processor=self.processor,
                threshold=self.config.validation_threshold,
            )

        # Initialize chains
        self.feature_chain = FeatureExtractionChain(
            agent=self.feature_agent, memory_type=self.config.memory_type
        )

        if self.config.enable_validation:
            self.validation_chain = ValidationChain(
                agent=self.validation_agent, memory_type=self.config.memory_type
            )

    def extract_features(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        task_description: str,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract features from multimodal data using agentic processing.

        Args:
            data: Input data (single sample or batch)
            task_description: Description of the feature extraction task
            validate: Whether to use validation agents

        Returns:
            Dictionary containing extracted features and metadata
        """
        if isinstance(data, dict):
            data = [data]

        results = []

        for sample in data:
            # Process multimodal data
            processed_data = self.processor.process(sample)

            # Extract features using feature agent
            features = self.feature_chain.run(
                input_data=processed_data, task_description=task_description
            )

            # Validate features if enabled
            if validate and self.validation_chain:
                validation_result = self.validation_chain.run(
                    features=features,
                    original_data=processed_data,
                    task_description=task_description,
                )

                if validation_result["is_valid"]:
                    features["validation_score"] = validation_result["score"]
                    features["validation_feedback"] = validation_result["feedback"]
                else:
                    # Refine features based on validation feedback
                    refined_features = self.feature_chain.run(
                        input_data=processed_data,
                        task_description=task_description,
                        feedback=validation_result["feedback"],
                    )
                    features.update(refined_features)
                    features["validation_score"] = validation_result["score"]
                    features["validation_feedback"] = validation_result["feedback"]

            results.append(features)

        return {
            "features": results,
            "metadata": {
                "num_samples": len(results),
                "model_name": self.config.model_name,
                "validation_enabled": validate and self.validation_chain is not None,
                "device": str(self.device),
            },
        }

    def train_adapter(
        self,
        training_data: List[Dict[str, Any]],
        task_description: str,
        epochs: int = 3,
        learning_rate: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Train the lightweight adapter on task-specific data.

        Args:
            training_data: Training samples
            task_description: Description of the training task
            epochs: Number of training epochs
            learning_rate: Learning rate for training

        Returns:
            Training metrics and results
        """
        return self.adapter.train(
            training_data=training_data,
            task_description=task_description,
            epochs=epochs,
            learning_rate=learning_rate,
        )

    def save_model(self, path: Union[str, Path]):
        """Save the trained model and adapters."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save adapter
        self.adapter.save(path / "adapter")

        # Save chain states
        if self.feature_chain:
            self.feature_chain.save_memory(path / "feature_memory")

        if self.validation_chain:
            self.validation_chain.save_memory(path / "validation_memory")

    def load_model(self, path: Union[str, Path]):
        """Load a previously saved model and adapters."""
        path = Path(path)

        # Load adapter
        self.adapter.load(path / "adapter")

        # Load chain states
        if self.feature_chain and (path / "feature_memory").exists():
            self.feature_chain.load_memory(path / "feature_memory")

        if self.validation_chain and (path / "validation_memory").exists():
            self.validation_chain.load_memory(path / "validation_memory")
