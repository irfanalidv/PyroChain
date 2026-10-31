"""Tests for agents, memory, CLI, and e-commerce utilities (tiny offline GPT-2)."""

import json

import pytest
import torch
from click.testing import CliRunner
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool

from pyrochain import PyroChain, PyroChainConfig
from pyrochain.agents.base import ConversationMemory, SequentialToolExecutor
from pyrochain.cli import _to_jsonable, main
from pyrochain.processors.text import TextProcessor
from pyrochain.utils import EcommerceFeatureExtractor, ProductAnalyzer, ProductData


def _product(pid="p1", title="wireless headphones", price=199.99, category="electronics"):
    return ProductData(
        product_id=pid,
        title=title,
        description="great battery life",
        price=price,
        category=category,
        brand="acme",
        images=[],
        attributes={"color": "black"},
        reviews=[{"rating": 5, "text": "great"}, {"rating": 2, "text": "bad"}],
        inventory=12,
    )


@pytest.fixture
def chain():
    return PyroChain(PyroChainConfig(model_name="tiny-gpt2", device="cpu", adapter_rank=4))


# ---------------------------------------------------------------- memory / executor


def test_conversation_memory_is_bounded_and_clearable():
    memory = ConversationMemory(max_messages=3)
    for i in range(5):
        memory.add_message(HumanMessage(content=str(i)))
    assert [m.content for m in memory.messages] == ["2", "3", "4"]
    memory.chat_memory.add_message(AIMessage(content="x"))  # legacy access path
    assert memory.messages[-1].content == "x"
    memory.clear()
    assert memory.messages == []


def test_sequential_tool_executor_runs_tools_in_order_and_records_memory():
    calls = []
    tools = [
        Tool(name="a", description="first", func=lambda q: calls.append("a") or "A-out"),
        Tool(name="b", description="second", func=lambda q: calls.append("b") or "B-out"),
    ]
    memory = ConversationMemory()
    transcript = SequentialToolExecutor(tools, memory).invoke("hello")
    assert calls == ["a", "b"]
    assert "Action: a\nObservation: A-out" in transcript
    assert transcript.splitlines()[-1].startswith("Final Answer:")
    assert [type(m) for m in memory.messages] == [HumanMessage, AIMessage]


# ---------------------------------------------------------------- validation


def test_validation_failure_triggers_refinement(chain):
    chain.validation_agent.threshold = 1.01  # impossible to pass
    result = chain.extract_features({"text": "great product"}, "task")
    features = result["features"][0]
    assert "failed validation" in features["validation_feedback"]
    assert features["validation_score"] < 1.01


def test_validation_metrics(chain):
    results = [
        {"score": 0.9, "is_valid": True},
        {"score": 0.5, "is_valid": False},
    ]
    metrics = chain.validation_agent.get_validation_metrics(results)
    assert metrics["valid_samples"] == 1 and metrics["average_score"] == pytest.approx(0.7)


# ---------------------------------------------------------------- text processor


def test_text_processor_fallback_is_deterministic_and_normalised():
    proc = TextProcessor(device=torch.device("cpu"))
    a = proc.get_features({"text": "great wireless headphones"})
    b = proc.get_features({"text": "great wireless headphones"})
    c = proc.get_features({"text": "terrible shirt"})
    assert a.shape == (1, 384)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.norm().item() == pytest.approx(1.0, abs=1e-5)
    assert proc.extract_sentiment("great great bad")["positive"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------- e-commerce


def test_ecommerce_feature_extraction(chain):
    out = EcommerceFeatureExtractor(chain).extract_product_features(_product(), "recommendation")
    assert out["product_metadata"]["product_id"] == "p1"
    assert "price_features" in out["ecommerce_features"]
    assert out["features"][0]["text_features"].shape == (1, 32)


def test_product_similarity(chain):
    analyzer = ProductAnalyzer(chain)
    same = analyzer.analyze_product_similarity(_product("a"), _product("b"))
    diff = analyzer.analyze_product_similarity(
        _product("a"), _product("c", title="organic cotton shirt", price=9.99, category="clothing")
    )
    assert 0.0 <= diff["overall_similarity"] <= same["overall_similarity"] <= 1.0 + 1e-6


def test_feature_visualizer_is_lazy():
    import pyrochain.utils as utils

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"pyrochain\[viz\]"):
            utils.FeatureVisualizer  # noqa: B018
    else:
        assert utils.FeatureVisualizer.__name__ == "FeatureVisualizer"


# ---------------------------------------------------------------- CLI


def test_to_jsonable_converts_tensors():
    assert _to_jsonable({"t": torch.tensor([[1.0, 2.0]]), "n": torch.tensor(3)}) == {
        "t": [[1.0, 2.0]],
        "n": 3,
    }


def test_cli_version():
    result = CliRunner().invoke(main, ["version"])
    assert result.exit_code == 0 and "PyroChain version" in result.output


def test_cli_extract_writes_json(tmp_path):
    inp = tmp_path / "in.json"
    out = tmp_path / "out.json"
    inp.write_text(json.dumps({"text": "great wireless headphones", "price": 10}))
    result = CliRunner().invoke(
        main,
        [
            "extract",
            "--input-file",
            str(inp),
            "--task",
            "t",
            "--output-file",
            str(out),
            "--model",
            "tiny-gpt2",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["metadata"]["num_samples"] == 1
    assert len(data["features"][0]["text_features"][0]) == 32


def test_cli_train_and_predict(tmp_path):
    train = tmp_path / "train.json"
    train.write_text(json.dumps([{"text": "great headphones"}, {"text": "bad shirt"}] * 2))
    model_dir = tmp_path / "model"
    runner = CliRunner()
    r1 = runner.invoke(
        main,
        [
            "train",
            "--training-file",
            str(train),
            "--task",
            "lm",
            "--epochs",
            "1",
            "--output-dir",
            str(model_dir),
            "--model",
            "tiny-gpt2",
            "--device",
            "cpu",
        ],
    )
    assert r1.exit_code == 0, r1.output
    assert "Training completed!" in r1.output
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps({"text": "great"}))
    r2 = runner.invoke(
        main,
        [
            "predict",
            "--model-dir",
            str(model_dir),
            "--input-file",
            str(inp),
            "--task",
            "t",
            "--model",
            "tiny-gpt2",
            "--device",
            "cpu",
        ],
    )
    assert r2.exit_code == 0, r2.output
    assert json.loads(r2.output[r2.output.index("{") :])["metadata"]["num_samples"] == 1
