"""Chains (memory, history, metrics) and multimodal processing (fusion, image inputs)."""

import numpy as np
import pytest
import torch
from langchain_core.messages import HumanMessage
from PIL import Image

from pyrochain import PyroChain, PyroChainConfig
from pyrochain.processors import MultimodalProcessor

CPU = torch.device("cpu")


@pytest.fixture
def chain():
    return PyroChain(PyroChainConfig(model_name="tiny-gpt2", device="cpu", adapter_rank=4))


class TestFeatureExtractionChain:
    def test_batch_history_and_metrics(self, chain):
        fc = chain.feature_chain
        fc.clear_memory()
        records = [chain.processor.process({"text": t}) for t in ("good shirt", "bad movie")]
        results = fc.batch_extract_features(records, "task")
        assert len(results) == 2
        assert all(r["chain_metadata"]["iteration"] == 0 for r in results)

        history = fc.get_extraction_history()
        assert len(history) == 2 and "Feature Extraction Result" in history[0]["output"]
        metrics = fc.get_chain_metrics()
        assert metrics["total_extractions"] == 2 and metrics["success_rate"] == 1.0

        state = fc.get_chain_state()
        assert state["agent_type"] == "FeatureAgent"
        assert state["memory_summary"]["num_messages"] == len(fc.get_memory_messages())

    def test_refinement_feedback_and_reset(self, chain):
        fc = chain.feature_chain
        assert fc._check_refinement_needed({"raw_features": None})
        assert fc._check_refinement_needed(
            {"raw_features": torch.zeros(1, 0), "metadata": {"a": 1}}
        )
        assert not fc._check_refinement_needed(
            {"raw_features": torch.ones(1, 2), "metadata": {"a": 1}}
        )
        assert "No raw features" in fc._generate_refinement_feedback({"raw_features": None})
        fc._add_result_to_memory({"raw_features": None}, 0)  # must not raise on None
        fc.extract_features_with_context({"text": "x"}, "t", context_data=[{"text": "y"}])
        fc.update_memory([HumanMessage(content="note")])
        fc.reset_chain()
        assert fc.get_memory_messages() == []

    def test_empty_metrics(self, chain):
        chain.feature_chain.clear_memory()
        assert chain.feature_chain.get_chain_metrics() == {"total_extractions": 0}


class TestValidationChain:
    def test_report_and_metrics_parse_history(self, chain):
        vc = chain.validation_chain
        vc.clear_memory()
        data = chain.processor.process({"text": "great headphones"})
        feats = chain.feature_chain.run(data, "t")
        vc.batch_validate_features([feats, feats], [data, data], "t")
        vc.validate_with_historical_context(feats, data, "t", historical_validations=[{}])

        metrics = vc.get_validation_metrics()
        assert metrics["total_validations"] == 3
        assert metrics["valid_count"] == 3
        assert metrics["average_score"] == pytest.approx(0.91)
        report = vc.generate_validation_report()
        assert "Run more validations to get better statistics" in report["recommendations"]
        assert len(report["recent_validations"]) == 3

    def test_mismatched_batch_lengths_raise(self, chain):
        with pytest.raises(ValueError):
            chain.validation_chain.batch_validate_features([{}], [], "t")


class TestMultimodalProcessor:
    @pytest.mark.parametrize("method", ["concat", "add", "multiply", "attention", "unknown"])
    def test_fusion_methods(self, chain, method):
        proc = MultimodalProcessor(adapter=chain.adapter, device=CPU, fusion_method=method)
        record = {"text": "red shirt", "image": Image.new("RGB", (16, 16), (200, 0, 0))}
        fused = proc.process(record)["multimodal_features"]
        text_dim = proc.text_processor.get_features(record).shape[-1]
        image_dim = proc.image_processor.feature_dim
        expected = (
            text_dim + image_dim if method in ("concat", "unknown") else max(text_dim, image_dim)
        )
        assert fused.shape == (1, expected)
        assert torch.isfinite(fused).all()

    def test_image_inputs_as_array_and_tensor(self, chain):
        proc = chain.processor
        arr = np.full((8, 8, 3), 0.5, dtype=np.float32)
        tensor = torch.rand(3, 8, 8, generator=torch.Generator().manual_seed(0))
        for img in (arr, tensor):
            out = proc.process({"photo": img})
            assert "image" in out["modalities"]

    def test_structured_features_and_summary(self, chain):
        record = {"title": "great cotton shirt", "image": Image.new("RGB", (20, 10), (0, 0, 255))}
        structured = chain.processor.extract_structured_features(record)
        assert structured["text_features"]["title"]["sentiment"]["positive"] > 0
        assert structured["image_features"]["image"]["dimensions"] == (20, 10)
        summary = chain.processor.get_feature_summary(record)
        assert summary["image_summary"]["num_image_fields"] == 1
        assert summary["modalities"] == ["text", "image"]
        assert summary["multimodal_summary"]["fusion_method"] == "concat"

    def test_keywords(self, chain):
        words = chain.processor.text_processor.extract_keywords("battery battery life is great")
        assert words[0] == "battery"
