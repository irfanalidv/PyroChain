"""
Product catalog: "similar items" and clustering from text, image, price and brand features.

Shows:
- ``ProductData`` records with images (solid-colour swatches generated here with Pillow,
  standing in for product photos).
- ``ProductAnalyzer.find_similar_products``: a weighted mix of text-embedding, image, price,
  category, brand and review similarity.
- ``ProductAnalyzer.cluster_products``: k-means over the fused multimodal feature vectors.

Data: ``examples/assets/listings.csv`` (8 example listings). Needs: ``pip install pyrochain``.
No API key. The first run downloads ``microsoft/DialoGPT-small`` (about 350 MB).

Run: python examples/product_catalog_similarity.py
"""

import csv
from pathlib import Path

from PIL import Image
from transformers.utils import logging as hf_logging

from pyrochain import PyroChain, PyroChainConfig
from pyrochain.utils import ProductAnalyzer, ProductData

LISTINGS = Path(__file__).parent / "assets" / "listings.csv"
# A sentence-embedding encoder separates product texts far better than DialoGPT's mean-pooled
# states, which are nearly parallel for any input (cosine similarity close to 1.0).
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLORS = {
    "black": (20, 20, 20),
    "blue": (30, 80, 200),
    "red": (200, 30, 40),
    "green": (40, 140, 60),
    "silver": (190, 190, 200),
}


def load_catalog() -> list[ProductData]:
    with LISTINGS.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        ProductData(
            product_id=r["product_id"],
            title=r["title"],
            description=r["description"],
            price=float(r["price"]),
            category=r["category"],
            brand=r["brand"],
            images=[Image.new("RGB", (64, 64), COLORS[r["color"]])],
            attributes={"color": r["color"]},
            reviews=[{"rating": float(r["rating"])}],
            inventory=10,
        )
        for r in rows
    ]


def main() -> None:
    hf_logging.set_verbosity_error()
    catalog = load_catalog()
    analyzer = ProductAnalyzer(PyroChain(PyroChainConfig(model_name=MODEL, device="cpu")))

    target = catalog[0]
    print(f"Most similar to {target.product_id} ({target.title}):")
    for match in analyzer.find_similar_products(target, catalog, top_k=3):
        sim = match["similarity"]
        parts = ", ".join(f"{k} {v:.2f}" for k, v in sim["feature_similarities"].items())
        print(
            f"  {match['product'].product_id:6} overall {sim['overall_similarity']:.2f}  ({parts})"
        )

    clusters = analyzer.cluster_products(catalog, n_clusters=3)
    print("\nk-means clusters (k=3) over fused feature vectors:")
    titles = {p.product_id: p.title for p in catalog}
    for label, ids in sorted(clusters["clusters"].items()):
        print(f"  cluster {label}: " + "; ".join(titles[i] for i in ids))


if __name__ == "__main__":
    main()
