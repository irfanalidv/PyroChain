"""
Build a model-ready feature matrix from product listings (text + image + tabular).

Shows:
- Multimodal records: text fields plus an image per listing.
- The ``fusion_method`` options of ``MultimodalProcessor`` and the vector size each produces.
- E-commerce tabular features (price buckets, stable category/brand codes, ratings) from
  ``EcommerceFeatureExtractor``, joined with the fused embedding into one NumPy matrix.
- ``FeatureMetrics`` sanity checks, and saving the matrix to ``.npz`` for a downstream model.

Data: ``examples/assets/listings.csv``. Needs: ``pip install pyrochain``. No API key. The first
run downloads ``microsoft/DialoGPT-small`` (about 350 MB). Output goes to a temporary folder.

Run: python examples/listing_feature_matrix.py
"""

import csv
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from transformers.utils import logging as hf_logging

from pyrochain import MultimodalProcessor, PyroChain, PyroChainConfig
from pyrochain.utils import EcommerceFeatureExtractor, FeatureMetrics, ProductData

LISTINGS = Path(__file__).parent / "assets" / "listings.csv"
COLORS = {
    "black": (20, 20, 20),
    "blue": (30, 80, 200),
    "red": (200, 30, 40),
    "green": (40, 140, 60),
    "silver": (190, 190, 200),
}


def main() -> None:
    hf_logging.set_verbosity_error()
    with LISTINGS.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    images = [Image.new("RGB", (64, 64), COLORS[r["color"]]) for r in rows]
    records = [
        {"title": r["title"], "description": r["description"], "image": img}
        for r, img in zip(rows, images, strict=True)
    ]

    chain = PyroChain(PyroChainConfig(device="cpu"))
    print("Fused vector size per fusion method:")
    for method in ("concat", "add", "multiply", "attention"):
        proc = MultimodalProcessor(chain.adapter, chain.device, fusion_method=method)
        print(f"  {method:9} {proc.get_features(records[0]).shape[-1]}")

    result = chain.extract_features(records, "Extract features for product search ranking")
    embeddings = np.vstack([f["raw_features"].cpu().numpy() for f in result["features"]])
    scores = [round(f["validation_score"], 2) for f in result["features"]]
    print(f"\nEmbeddings: {embeddings.shape}, validation scores: {sorted(set(scores))}")

    extractor = EcommerceFeatureExtractor(chain)
    tabular = []
    for r in rows:
        product = ProductData(
            product_id=r["product_id"],
            title=r["title"],
            description=r["description"],
            price=float(r["price"]),
            category=r["category"],
            brand=r["brand"],
            images=[],
            attributes={},
            reviews=[{"rating": float(r["rating"])}],
            inventory=10,
        )
        ec = extractor.extract_tabular_features(product)
        tabular.append(
            [
                ec["price_features"]["price_log"],
                ec["price_features"]["price_category"],
                ec["category_features"]["category_encoded"],
                ec["brand_features"]["brand_encoded"],
                ec["review_features"]["avg_rating"],
            ]
        )
    tab = np.asarray(tabular, dtype=np.float32)
    matrix = np.hstack([embeddings, tab])
    print(f"Tabular block: {tab.shape}; final matrix: {matrix.shape}")

    quality = FeatureMetrics().calculate_feature_quality_metrics(embeddings)
    print(
        f"Embedding stats: mean {quality['mean']:.3f}, std {quality['std']:.3f}, "
        f"zero ratio {quality['zero_ratio']:.2f}"
    )

    out = Path(tempfile.mkdtemp()) / "listing_features.npz"
    np.savez(out, X=matrix, product_id=np.array([r["product_id"] for r in rows]))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
