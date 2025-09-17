"""
Tests for PyroChain core functionality.
"""

import pytest
import torch
from pyrochain import PyroChain, PyroChainConfig


class TestPyroChain:
    """Test cases for PyroChain core functionality."""
    
    def test_initialization(self):
        """Test PyroChain initialization."""
        config = PyroChainConfig(
            model_name="google/gemma-2b",
            device="cpu",
            adapter_rank=16
        )
        
        pyrochain = PyroChain(config)
        
        assert pyrochain.config.model_name == "google/gemma-2b"
        assert pyrochain.device.type == "cpu"
        assert pyrochain.adapter is not None
        assert pyrochain.processor is not None
        assert pyrochain.feature_agent is not None
        assert pyrochain.validation_agent is not None
        assert pyrochain.feature_chain is not None
        assert pyrochain.validation_chain is not None
    
    def test_device_setup(self):
        """Test device setup."""
        config = PyroChainConfig(device="cpu")
        pyrochain = PyroChain(config)
        assert pyrochain.device.type == "cpu"
        
        config = PyroChainConfig(device="auto")
        pyrochain = PyroChain(config)
        # Should default to CPU if CUDA/MPS not available
        assert pyrochain.device.type in ["cpu", "cuda", "mps"]
    
    def test_feature_extraction(self):
        """Test basic feature extraction."""
        config = PyroChainConfig(device="cpu")
        pyrochain = PyroChain(config)
        
        sample_data = {
            "text": "This is a test product",
            "title": "Test Product",
            "price": 99.99
        }
        
        task_description = "Extract features for product recommendation"
        
        results = pyrochain.extract_features(
            data=sample_data,
            task_description=task_description,
            validate=True
        )
        
        assert "features" in results
        assert "metadata" in results
        assert results["metadata"]["num_samples"] == 1
        assert results["metadata"]["validation_enabled"] is True
    
    def test_batch_processing(self):
        """Test batch processing."""
        config = PyroChainConfig(device="cpu")
        pyrochain = PyroChain(config)
        
        batch_data = [
            {"text": "Product 1", "title": "Title 1"},
            {"text": "Product 2", "title": "Title 2"}
        ]
        
        task_description = "Extract features for batch processing"
        
        results = pyrochain.extract_features(
            data=batch_data,
            task_description=task_description,
            validate=False
        )
        
        assert "features" in results
        assert len(results["features"]) == 2
        assert results["metadata"]["num_samples"] == 2
    
    def test_model_save_load(self, tmp_path):
        """Test model saving and loading."""
        config = PyroChainConfig(device="cpu")
        pyrochain = PyroChain(config)
        
        # Save model
        model_path = tmp_path / "test_model"
        pyrochain.save_model(model_path)
        
        # Check that files were created
        assert (model_path / "adapter").exists()
        
        # Create new instance and load model
        new_pyrochain = PyroChain(config)
        new_pyrochain.load_model(model_path)
        
        # Both should have the same adapter type
        assert type(pyrochain.adapter) == type(new_pyrochain.adapter)


class TestPyroChainConfig:
    """Test cases for PyroChainConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = PyroChainConfig()
        
        assert config.model_name == "google/gemma-2b"
        assert config.adapter_rank == 16
        assert config.max_length == 512
        assert config.device == "auto"
        assert config.memory_type == "conversation_buffer"
        assert config.enable_validation is True
        assert config.validation_threshold == 0.8
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PyroChainConfig(
            model_name="custom-model",
            adapter_rank=32,
            max_length=1024,
            device="cuda",
            memory_type="conversation_summary",
            enable_validation=False,
            validation_threshold=0.9
        )
        
        assert config.model_name == "custom-model"
        assert config.adapter_rank == 32
        assert config.max_length == 1024
        assert config.device == "cuda"
        assert config.memory_type == "conversation_summary"
        assert config.enable_validation is False
        assert config.validation_threshold == 0.9
