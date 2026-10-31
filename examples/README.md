# PyroChain examples

Runnable scripts, each with a docstring saying what it shows and what it needs. None needs an
API key. The first run downloads `microsoft/DialoGPT-small` (about 350 MB) from the Hugging Face
Hub (`product_catalog_similarity.py` uses `sentence-transformers/all-MiniLM-L6-v2`, about
90 MB); later runs work offline. Sample data lives in [`assets/`](assets/).

| Script | Use case |
| --- | --- |
| [`quickstart.py`](quickstart.py) | Extract and validate features for one product record |
| [`review_sentiment_classifier.py`](review_sentiment_classifier.py) | Text features into a scikit-learn sentiment classifier, with cross-validated accuracy |
| [`product_catalog_similarity.py`](product_catalog_similarity.py) | "Similar items" and k-means clusters over text, image, price and brand features |
| [`listing_feature_matrix.py`](listing_feature_matrix.py) | Fused text + image embeddings joined with tabular e-commerce features, saved as `.npz` |
| [`support_adapter_finetune.py`](support_adapter_finetune.py) | Fine-tune the LoRA adapter on domain text, then save and reload it |

```bash
pip install pyrochain
python examples/quickstart.py
```
