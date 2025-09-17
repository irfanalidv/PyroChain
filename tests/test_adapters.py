"""
Tests for PyroChain adapters.
"""

import pytest
import torch
from pyrochain.adapters import LightweightAdapter, LoRAAdapter


class TestLightweightAdapter:
    """Test cases for LightweightAdapter."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        device = torch.device("cpu")
        adapter = LightweightAdapter(
            model_name="google/gemma-2b",
            device=device,
            adapter_type="lora",
            rank=16
        )
        
        assert adapter.model_name == "google/gemma-2b"
        assert adapter.device == device
        assert adapter.adapter_type == "lora"
        assert adapter.rank == 16
        assert adapter.adapter is not None
    
    def test_text_encoding(self):
        """Test text encoding."""
        device = torch.device("cpu")
        adapter = LightweightAdapter(
            model_name="google/gemma-2b",
            device=device
        )
        
        text = "This is a test text"
        encoding = adapter.encode_text(text)
        
        assert "input_ids" in encoding
        assert "attention_mask" in encoding
        assert encoding["input_ids"].device == device
        assert encoding["attention_mask"].device == device
    
    def test_embeddings(self):
        """Test embedding extraction."""
        device = torch.device("cpu")
        adapter = LightweightAdapter(
            model_name="google/gemma-2b",
            device=device
        )
        
        text = "This is a test text"
        embeddings = adapter.get_embeddings(text)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.device == device
        assert embeddings.dim() == 1  # Should be 1D vector
    
    def test_save_load(self, tmp_path):
        """Test adapter saving and loading."""
        device = torch.device("cpu")
        adapter = LightweightAdapter(
            model_name="google/gemma-2b",
            device=device
        )
        
        # Save adapter
        adapter_path = tmp_path / "adapter"
        adapter.save(adapter_path)
        
        # Check that files were created
        assert (adapter_path / "config.json").exists()
        
        # Create new adapter and load
        new_adapter = LightweightAdapter(
            model_name="google/gemma-2b",
            device=device
        )
        new_adapter.load(adapter_path)
        
        # Both should have the same configuration
        assert adapter.rank == new_adapter.rank
        assert adapter.alpha == new_adapter.alpha


class TestLoRAAdapter:
    """Test cases for LoRAAdapter."""
    
    def test_initialization(self):
        """Test LoRA adapter initialization."""
        device = torch.device("cpu")
        adapter = LoRAAdapter(
            model_name="google/gemma-2b",
            device=device,
            rank=16,
            alpha=16.0
        )
        
        assert adapter.model_name == "google/gemma-2b"
        assert adapter.device == device
        assert adapter.rank == 16
        assert adapter.alpha == 16.0
        assert adapter.lora_layers is not None
    
    def test_forward_pass(self):
        """Test forward pass through LoRA adapter."""
        device = torch.device("cpu")
        adapter = LoRAAdapter(
            model_name="google/gemma-2b",
            device=device,
            rank=16
        )
        
        # Create dummy input
        batch_size = 2
        seq_length = 10
        hidden_size = 768  # Typical hidden size
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device)
        
        # Forward pass
        output = adapter.forward(input_ids, attention_mask)
        
        assert isinstance(output, torch.Tensor)
        assert output.device == device
        assert output.shape[0] == batch_size
        assert output.shape[1] == seq_length
    
    def test_trainable_parameters(self):
        """Test trainable parameters count."""
        device = torch.device("cpu")
        adapter = LoRAAdapter(
            model_name="google/gemma-2b",
            device=device,
            rank=16
        )
        
        trainable_params = adapter.get_trainable_parameters()
        assert trainable_params > 0
        assert isinstance(trainable_params, int)
    
    def test_training_step(self):
        """Test training step."""
        device = torch.device("cpu")
        adapter = LoRAAdapter(
            model_name="google/gemma-2b",
            device=device,
            rank=16
        )
        
        # Create dummy batch
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10), device=device),
            "attention_mask": torch.ones(2, 10, device=device),
            "labels": torch.randint(0, 1000, (2, 10), device=device)
        }
        
        # Training step
        metrics = adapter.train_step(batch)
        
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
    
    def test_save_load(self, tmp_path):
        """Test LoRA adapter saving and loading."""
        device = torch.device("cpu")
        adapter = LoRAAdapter(
            model_name="google/gemma-2b",
            device=device,
            rank=16,
            alpha=16.0
        )
        
        # Save adapter
        adapter_path = tmp_path / "lora_adapter"
        adapter.save(adapter_path)
        
        # Check that files were created
        assert (adapter_path / "config.json").exists()
        assert (adapter_path / "lora_weights.pt").exists()
        
        # Create new adapter and load
        new_adapter = LoRAAdapter(
            model_name="google/gemma-2b",
            device=device
        )
        new_adapter.load(adapter_path)
        
        # Both should have the same configuration
        assert adapter.rank == new_adapter.rank
        assert adapter.alpha == new_adapter.alpha
