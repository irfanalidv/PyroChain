"""Tests for PyroChain adapters (tiny offline GPT-2, see conftest.py)."""

import torch

from pyrochain.adapters import LightweightAdapter, LoRAAdapter

CPU = torch.device("cpu")


class TestLightweightAdapter:
    def test_initialization(self):
        adapter = LightweightAdapter(
            model_name="tiny-gpt2", device=CPU, adapter_type="lora", rank=4
        )
        assert adapter.model_name == "tiny-gpt2"
        assert adapter.device == CPU
        assert adapter.adapter_type == "lora"
        assert adapter.rank == 4
        assert adapter.adapter is not None

    def test_unknown_adapter_type(self):
        import pytest

        with pytest.raises(ValueError, match="Unsupported adapter type"):
            LightweightAdapter(model_name="tiny-gpt2", device=CPU, adapter_type="prefix")

    def test_text_encoding(self):
        adapter = LightweightAdapter(model_name="tiny-gpt2", device=CPU)
        encoding = adapter.encode_text("This is a test text")
        assert "input_ids" in encoding and "attention_mask" in encoding
        assert encoding["input_ids"].device == CPU

    def test_embeddings(self, tiny_models):
        adapter = LightweightAdapter(model_name="tiny-gpt2", device=CPU)
        embeddings = adapter.get_embeddings("This is a test text")
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (1, tiny_models["hidden"])

    def test_train_reduces_loss(self):
        torch.manual_seed(0)
        adapter = LightweightAdapter(model_name="tiny-gpt2", device=CPU, rank=4)
        data = [{"text": "great wireless headphones"}, {"text": "organic cotton shirt"}] * 4
        result = adapter.train(
            data, "language modelling", epochs=3, learning_rate=5e-2, validation_split=0.25
        )
        history = result["training_history"]
        assert len(history) == 3
        assert history[-1]["train_loss"] < history[0]["train_loss"]
        assert "val_loss" in history[0]
        assert result["total_parameters"] > 0
        assert len(data) == 8  # caller's list is not mutated/shuffled away

    def test_save_load(self, tmp_path):
        adapter = LightweightAdapter(model_name="tiny-gpt2", device=CPU, rank=4)
        adapter.save(tmp_path / "adapter")
        assert (tmp_path / "adapter" / "config.json").exists()

        new_adapter = LightweightAdapter(model_name="tiny-gpt2", device=CPU, rank=4)
        new_adapter.load(tmp_path / "adapter")
        for (k1, v1), (k2, v2) in zip(
            adapter.adapter.lora_layers.state_dict().items(),
            new_adapter.adapter.lora_layers.state_dict().items(),
        ):
            assert k1 == k2 and torch.equal(v1, v2)


class TestLoRAAdapter:
    def test_initialization_and_fallback_layer(self):
        # GPT-2 attention uses Conv1D, so no default target matches: a residual layer is created.
        adapter = LoRAAdapter(model_name="tiny-gpt2", device=CPU, rank=4, alpha=8.0)
        assert adapter.rank == 4 and adapter.alpha == 8.0
        assert list(adapter.lora_layers.keys()) == ["final_hidden"]

    def test_targets_linear_modules_with_dotted_names(self):
        # GPT-2's MLP projections are Conv1D too; use a model with nn.Linear via a custom target.
        adapter = LoRAAdapter(model_name="tiny-gpt2", device=CPU, rank=4)
        linear = torch.nn.Linear(32, 32)
        adapter.model.h[0].extra_proj = linear
        adapter.target_modules = ["extra_proj"]
        adapter._setup_lora_layers()
        assert list(adapter.lora_layers.keys()) == ["h__0__extra_proj"]

    def test_base_model_frozen(self):
        adapter = LoRAAdapter(model_name="tiny-gpt2", device=CPU, rank=4)
        assert all(not p.requires_grad for p in adapter.model.parameters())
        assert all(p.requires_grad for p in adapter.lora_layers.parameters())

    def test_forward_pass(self, tiny_models):
        adapter = LoRAAdapter(model_name="tiny-gpt2", device=CPU, rank=4)
        input_ids = torch.randint(0, 50, (2, 10))
        output = adapter.forward(input_ids, torch.ones(2, 10))
        assert output.shape == (2, 10, tiny_models["hidden"])

    def test_trainable_parameters(self):
        adapter = LoRAAdapter(model_name="tiny-gpt2", device=CPU, rank=4)
        n = adapter.get_trainable_parameters()
        assert isinstance(n, int) and n == 2 * 32 * 4

    def test_training_step_returns_differentiable_loss(self):
        adapter = LoRAAdapter(model_name="tiny-gpt2", device=CPU, rank=4)
        batch = {
            "input_ids": torch.randint(0, 50, (2, 10)),
            "attention_mask": torch.ones(2, 10),
            "labels": torch.randint(0, 50, (2, 10)),
        }
        loss = adapter.train_step(batch)["loss"]
        assert loss.requires_grad and loss.item() > 0
        loss.backward()
        assert adapter.lora_layers["final_hidden"].lora_B.weight.grad is not None

    def test_save_load(self, tmp_path):
        adapter = LoRAAdapter(model_name="tiny-gpt2", device=CPU, rank=4, alpha=8.0)
        adapter.save(str(tmp_path / "lora"))
        assert (tmp_path / "lora" / "lora_weights.pt").exists()
        new_adapter = LoRAAdapter(model_name="tiny-gpt2", device=CPU)
        new_adapter.load(str(tmp_path / "lora"))
        assert (new_adapter.rank, new_adapter.alpha) == (4, 8.0)
