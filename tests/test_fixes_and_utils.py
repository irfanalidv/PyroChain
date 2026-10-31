"""Regression tests for the 0.3.0 fixes, plus metrics, image and visualisation coverage."""

import json
import logging
import subprocess
import sys

import numpy as np
import pytest
import torch
from click.testing import CliRunner
from PIL import Image

from pyrochain import PyroChain, PyroChainConfig
from pyrochain.adapters import LightweightAdapter
from pyrochain.cli import main
from pyrochain.processors.image import ImageProcessor
from pyrochain.utils import EcommerceFeatureExtractor, FeatureMetrics, ProductAnalyzer, ProductData
from pyrochain.utils.ecommerce import first_record, stable_bucket

CPU = torch.device("cpu")


@pytest.fixture
def chain():
    return PyroChain(PyroChainConfig(model_name="tiny-gpt2", device="cpu", adapter_rank=4))


def _product(pid, title, category="audio", brand="acme", price=100.0, images=None, **kw):
    return ProductData(
        product_id=pid,
        title=title,
        description=kw.get("description", "wireless battery life"),
        price=price,
        category=category,
        brand=brand,
        images=images or [],
        attributes=kw.get("attributes", {"color": "black"}),
        reviews=kw.get("reviews", [{"rating": 4, "text": "great"}]),
        inventory=kw.get("inventory", 5),
    )


def _image(color, size=(32, 24), mode="RGB"):
    return Image.new(mode, size, color)


# ------------------------------------------------------------------ validation


class TestValidation:
    def test_text_only_record_is_not_credited_with_image_features(self, chain):
        features = chain.extract_features({"text": "great product"}, "task")["features"][0]
        # completeness 1.0, quality 1.0, relevance 0.7 (no image) -> 0.91
        assert features["validation_score"] == pytest.approx(0.91)

    def test_text_and_image_record_scores_full_marks(self, chain):
        record = {"text": "red shirt", "image": _image((200, 30, 30))}
        assert chain.extract_features(record, "task")["features"][0]["validation_score"] == 1.0

    def test_empty_record_fails_and_is_refined_and_revalidated(self, chain):
        features = chain.extract_features({"price": 3.0}, "task")["features"][0]
        assert features["refined"] is True
        assert features["validation_score"] < chain.config.validation_threshold
        assert "No text or image features detected" in features["validation_feedback"]

    def test_additional_feedback_reflects_the_validated_features(self, chain):
        good = chain.extract_features({"text": "great product"}, "t", validate=False)
        record = good["features"][0]
        extra = chain.validation_chain._generate_additional_feedback({"score": 0.7}, record)
        assert extra == "Minor improvements needed"
        empty = chain.validation_chain._generate_additional_feedback({"score": 0.2}, {})
        assert "Ensure raw features are extracted" in empty


# ------------------------------------------------------------------ adapter training


class TestAdapterTraining:
    def test_single_sample_still_trains(self):
        adapter = LightweightAdapter(model_name="tiny-gpt2", device=CPU, rank=4)
        before = [p.clone() for p in adapter.adapter.lora_layers.parameters()]
        out = adapter.train([{"text": "great headphones battery life"}], "lm", epochs=1)
        after = list(adapter.adapter.lora_layers.parameters())
        assert out["training_history"][0]["train_loss"] > 0
        assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))

    def test_empty_or_textless_data_raises(self):
        adapter = LightweightAdapter(model_name="tiny-gpt2", device=CPU, rank=4)
        with pytest.raises(ValueError, match="empty"):
            adapter.train([], "lm")
        with pytest.raises(ValueError, match="text"):
            adapter.train([{"label": 1}], "lm")

    def test_split_is_seeded_and_does_not_touch_global_rng(self):
        adapter = LightweightAdapter(model_name="tiny-gpt2", device=CPU, rank=4)
        data = [{"text": str(i)} for i in range(10)]
        import random

        random.seed(123)
        expected_next = random.random()
        random.seed(123)
        a = adapter._split_data(data, 0.2, seed=7)
        b = adapter._split_data(data, 0.2, seed=7)
        assert a == b and len(a[0]) == 8 and len(a[1]) == 2
        assert random.random() == expected_next

    def test_training_logs_instead_of_printing(self, caplog, capsys):
        adapter = LightweightAdapter(model_name="tiny-gpt2", device=CPU, rank=4)
        with caplog.at_level(logging.INFO, logger="pyrochain.adapters.adapter"):
            adapter.train([{"text": "good shirt"}] * 3, "lm", epochs=1)
        assert "Epoch 1/1" in caplog.text
        assert "Epoch" not in capsys.readouterr().out

    def test_load_without_weights_raises(self, tmp_path):
        adapter = LightweightAdapter(model_name="tiny-gpt2", device=CPU, rank=4)
        adapter.save(tmp_path)
        (tmp_path / "lora_weights.pt").unlink()
        with pytest.raises(FileNotFoundError):
            adapter.load(tmp_path)


# ------------------------------------------------------------------ e-commerce


class TestEcommerce:
    def test_category_and_brand_encodings_are_stable_across_processes(self):
        code = (
            "from pyrochain.utils.ecommerce import stable_bucket;"
            "print(stable_bucket('electronics'), stable_bucket('acme'))"
        )
        runs = {
            subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                check=True,
                env={"PYTHONHASHSEED": seed, "PATH": ""},
            ).stdout
            for seed in ("1", "2")
        }
        assert len(runs) == 1
        assert runs.pop().split() == [str(stable_bucket("electronics")), str(stable_bucket("acme"))]

    def test_boolean_attributes_are_encoded_as_bools(self, chain):
        extractor = EcommerceFeatureExtractor(chain)
        feats = extractor._extract_attribute_features({"wireless": True, "weight": 2.5})
        assert feats["attr_wireless_bool"] == 1
        assert "attr_wireless_numeric" not in feats
        assert feats["attr_weight_numeric"] == 2.5

    def test_product_images_reach_the_image_processor(self, chain, tmp_path):
        path = tmp_path / "shoe.png"
        _image((10, 200, 10)).save(path)
        extractor = EcommerceFeatureExtractor(chain)
        result = extractor.extract_product_features(_product("p", "shoe", images=[str(path)]))
        assert first_record(result)["metadata"]["modalities"] == ["text", "image"]

    def test_similarity_uses_text_and_image_features(self, chain):
        analyzer = ProductAnalyzer(chain)
        a = _product("a", "wireless headphones", images=[_image((0, 0, 0))])
        b = _product("b", "wireless headphones", images=[_image((0, 0, 0))])
        out = analyzer.analyze_product_similarity(a, b)
        assert out["feature_similarities"]["text"] == pytest.approx(1.0, abs=1e-5)
        assert out["feature_similarities"]["image"] == pytest.approx(1.0, abs=1e-5)
        assert out["overall_similarity"] == pytest.approx(1.0, abs=1e-5)

    def test_find_similar_and_cluster_products(self, chain):
        analyzer = ProductAnalyzer(chain)
        catalog = [
            _product("h1", "wireless headphones", price=100),
            _product("h2", "wireless headphones", price=110),
            _product("s1", "organic cotton shirt", category="apparel", brand="zed", price=20),
            _product("s2", "organic cotton shirt", category="apparel", brand="zed", price=25),
        ]
        ranked = analyzer.find_similar_products(catalog[0], catalog, top_k=2)
        assert ranked[0]["product"].product_id == "h2"

        clusters = analyzer.cluster_products(catalog, n_clusters=2)
        groups = sorted(sorted(v) for v in clusters["clusters"].values())
        assert groups == [["h1", "h2"], ["s1", "s2"]]
        json.dumps(clusters)  # labels are plain ints, result is JSON-serialisable
        with pytest.raises(ValueError):
            analyzer.cluster_products(catalog, n_clusters=10)


# ------------------------------------------------------------------ images


class TestImageProcessor:
    @pytest.mark.parametrize(
        "mode,color", [("RGB", (255, 0, 0)), ("L", 128), ("RGBA", (0, 0, 255, 255))]
    )
    def test_extract_colors_handles_modes(self, mode, color):
        proc = ImageProcessor(device=CPU)
        colors = proc.extract_colors(_image(color, mode=mode), n_colors=3)
        assert len(colors) == 1  # a solid image has one colour
        assert colors[0]["hex"].startswith("#") and len(colors[0]["hex"]) == 7
        assert colors[0]["percentage"] == pytest.approx(1.0)

    def test_two_colour_image(self):
        img = _image((255, 0, 0), size=(20, 10))
        img.paste((0, 0, 255), (10, 0, 20, 10))
        colors = ImageProcessor(device=CPU).extract_colors(img, n_colors=2)
        assert {c["hex"] for c in colors} == {"#ff0000", "#0000ff"}

    def test_detect_objects_and_basic_features(self):
        img = _image((0, 0, 0), size=(100, 100))
        img.paste((255, 255, 255), (20, 20, 80, 80))
        proc = ImageProcessor(device=CPU)
        objects = proc.detect_objects(img)
        assert objects and objects[0]["bbox"][2] > 50
        processed = proc.process({"image": img})
        assert processed["image_features"].shape == (1, proc.feature_dim)
        assert processed["processed_images"]["image"]["aspect_ratio"] == 1.0


# ------------------------------------------------------------------ metrics


class TestFeatureMetrics:
    def test_quality_metrics_with_ground_truth(self):
        m = FeatureMetrics().calculate_feature_quality_metrics(
            torch.tensor([[0.0, 1.0, 2.0, 3.0]]), ground_truth=np.array([[0.0, 1.0, 2.0, 3.0]])
        )
        assert m["mean"] == pytest.approx(1.5)
        assert m["zero_ratio"] == pytest.approx(0.25)
        assert m["correlation"] == pytest.approx(1.0)
        assert m["mse"] == 0.0 and m["mae"] == 0.0

    def test_similarity_metrics_including_binary_jaccard(self):
        m = FeatureMetrics().calculate_similarity_metrics(
            np.array([1.0, 0.0, 1.0, 1.0]), np.array([1.0, 1.0, 0.0, 1.0])
        )
        assert m["jaccard_similarity"] == pytest.approx(0.5)
        assert m["cosine_similarity"] == pytest.approx(2 / 3)
        assert m["manhattan_distance"] == 2.0

    def test_validation_clustering_training_and_ecommerce_metrics(self):
        fm = FeatureMetrics()
        v = fm.calculate_validation_metrics(
            [{"score": 0.9, "is_valid": True}, {"score": 0.4, "is_valid": False}]
        )
        assert v["validation_rate"] == 0.5 and v["score_below_threshold"] == 0.5

        feats = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
        c = fm.calculate_clustering_metrics(feats, [0, 0, 1, 1])
        assert c["num_clusters"] == 2.0 and c["silhouette_score"] > 0.9

        history = [{"train_loss": x, "val_loss": x + 0.1} for x in (3.0, 2.0, 1.5, 1.2, 1.1, 1.0)]
        t = fm.calculate_training_metrics(history)
        assert t["train_loss_improvement"] == pytest.approx(2.0)
        assert "convergence_std" in t and "overfitting_indicator" in t

        products = [
            {
                "ecommerce_features": {
                    "price_features": {"price": p},
                    "category_features": {"category": c},
                    "review_features": {"avg_rating": r},
                }
            }
            for p, c, r in ((10.0, "a", 4.5), (30.0, "b", 3.0))
        ]
        e = fm.calculate_ecommerce_metrics(products)
        assert e["mean_price"] == 20.0 and e["num_categories"] == 2.0
        assert e["high_rated_products"] == 0.5

        report = fm.generate_comprehensive_report(
            {"raw_features": torch.ones(1, 4)}, validation_results=[{"score": 1.0}]
        )
        assert set(report) == {"feature_quality", "validation_metrics"}


# ------------------------------------------------------------------ visualisation


def test_feature_visualizer_draws_and_saves(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib

    matplotlib.use("Agg")
    from pyrochain.utils import FeatureVisualizer

    viz = FeatureVisualizer(figsize=(4, 3))
    feats = torch.randn(12, 6, generator=torch.Generator().manual_seed(0))
    viz.plot_feature_distribution(feats, save_path=str(tmp_path / "dist.png"))
    viz.plot_feature_embedding(feats, method="pca", save_path=str(tmp_path / "emb.png"))
    viz.plot_clustering_results(feats, [0] * 6 + [1] * 6, save_path=str(tmp_path / "clu.png"))
    viz.plot_training_history(
        [{"epoch": 1, "train_loss": 2.0}, {"epoch": 2, "train_loss": 1.0}],
        save_path=str(tmp_path / "hist.png"),
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "clu.png",
        "dist.png",
        "emb.png",
        "hist.png",
    ]


# ------------------------------------------------------------------ CLI


def test_cli_reports_bad_input_files(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    runner = CliRunner()
    result = runner.invoke(main, ["extract", "--input-file", str(bad), "--task", "t"])
    assert result.exit_code == 2 and "not valid JSON" in result.output
    missing = runner.invoke(main, ["extract", "--input-file", str(tmp_path / "x"), "--task", "t"])
    assert missing.exit_code == 2 and "file not found" in missing.output


def test_cli_ecommerce_command(tmp_path):
    product = {
        "product_id": "p1",
        "title": "wireless headphones",
        "description": "great battery life",
        "price": 99.0,
        "category": "audio",
        "brand": "acme",
        "images": [],
        "attributes": {"wireless": True},
        "reviews": [{"rating": 5, "text": "great"}],
        "inventory": 3,
    }
    path = tmp_path / "p.json"
    path.write_text(json.dumps(product))
    out = tmp_path / "o.json"
    result = CliRunner().invoke(
        main,
        [
            "ecommerce",
            "--product-file",
            str(path),
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
    assert data["ecommerce_features"]["attribute_features"]["attr_wireless_bool"] == 1
