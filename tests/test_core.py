"""End-to-end tests for PyroChain (tiny offline GPT-2, see conftest.py)."""

import numpy as np
import pytest
import torch
from PIL import Image

import pyrochain
from pyrochain import PyroChain, PyroChainConfig


@pytest.fixture
def chain():
    return PyroChain(PyroChainConfig(model_name="tiny-gpt2", device="cpu", adapter_rank=4))


def test_version_exposed():
    assert isinstance(pyrochain.__version__, str) and pyrochain.__version__.count(".") >= 2


class TestPyroChain:
    def test_initialization(self, chain):
        assert chain.config.model_name == "tiny-gpt2"
        assert chain.device.type == "cpu"
        for attr in (
            "adapter",
            "processor",
            "feature_agent",
            "validation_agent",
            "feature_chain",
            "validation_chain",
        ):
            assert getattr(chain, attr) is not None

    def test_device_auto(self):
        chain = PyroChain(PyroChainConfig(model_name="tiny-gpt2", device="auto"))
        assert chain.device.type in ["cpu", "cuda", "mps"]

    def test_validation_disabled(self):
        chain = PyroChain(PyroChainConfig(model_name="tiny-gpt2", enable_validation=False))
        assert chain.validation_agent is None and chain.validation_chain is None
        result = chain.extract_features({"text": "great product"}, "task")
        assert result["metadata"]["validation_enabled"] is False

    def test_feature_extraction(self, chain):
        sample = {"text": "This is a test product", "title": "Test Product", "price": 99.99}
        results = chain.extract_features(sample, "Extract features for product recommendation")

        assert results["metadata"]["num_samples"] == 1
        assert results["metadata"]["validation_enabled"] is True
        features = results["features"][0]
        # Text features come from the base model's hidden states (hidden size 32), not noise
        assert features["text_features"].shape == (1, 32)
        assert features["metadata"]["modalities"] == ["text"]
        assert 0.0 <= features["validation_score"] <= 1.0
        assert "Action: extract_text_features" in features["agent_reasoning"]

    def test_features_are_deterministic(self, chain):
        sample = {"text": "great wireless headphones"}
        a = chain.extract_features(sample, "t", validate=False)["features"][0]["text_features"]
        b = chain.extract_features(sample, "t", validate=False)["features"][0]["text_features"]
        assert torch.allclose(a, b)

    def test_multimodal_with_image(self, chain):
        image = Image.fromarray(np.full((32, 32, 3), 200, dtype=np.uint8))
        result = chain.extract_features({"text": "organic cotton shirt", "image": image}, "t")
        features = result["features"][0]
        assert set(features["metadata"]["modalities"]) == {"text", "image"}
        assert features["raw_features"].shape[-1] == 32 + 2048  # concat fusion

    def test_batch_processing(self, chain):
        batch = [
            {"text": "Product 1", "title": "Title 1"},
            {"text": "Product 2", "title": "Title 2"},
        ]
        results = chain.extract_features(batch, "batch", validate=False)
        assert len(results["features"]) == 2
        assert results["metadata"]["num_samples"] == 2

    def test_train_adapter(self, chain):
        data = [{"text": "great headphones"}, {"text": "bad shirt"}] * 2
        out = chain.train_adapter(data, "lm", epochs=1)
        assert len(out["training_history"]) == 1

    def test_model_save_load(self, chain, tmp_path):
        chain.extract_features({"text": "great product"}, "t")
        chain.save_model(tmp_path / "model")
        assert (tmp_path / "model" / "adapter" / "lora_weights.pt").exists()
        assert (tmp_path / "model" / "feature_memory" / "memory.json").exists()

        new_chain = PyroChain(PyroChainConfig(model_name="tiny-gpt2", device="cpu", adapter_rank=4))
        new_chain.load_model(tmp_path / "model")
        assert len(new_chain.feature_agent.memory.messages) == len(
            chain.feature_agent.memory.messages
        )


class TestPyroChainConfig:
    def test_default_config(self):
        config = PyroChainConfig()
        assert config.model_name == "microsoft/DialoGPT-small"
        assert config.adapter_rank == 16
        assert config.max_length == 512
        assert config.device == "auto"
        assert config.memory_type == "conversation_buffer"
        assert config.enable_validation is True
        assert config.validation_threshold == 0.8

    def test_custom_config(self):
        config = PyroChainConfig(
            model_name="custom-model",
            adapter_rank=32,
            max_length=1024,
            device="cuda",
            memory_type="conversation_summary",
            enable_validation=False,
            validation_threshold=0.9,
        )
        assert (config.model_name, config.adapter_rank, config.max_length) == (
            "custom-model",
            32,
            1024,
        )
        assert config.enable_validation is False and config.validation_threshold == 0.9
