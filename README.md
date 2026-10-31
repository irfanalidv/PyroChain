<div align="center">

# PyroChain 🔥

**Intelligent feature engineering with AI agents: PyTorch + LangChain for multimodal data.**

[![PyPI version](https://img.shields.io/pypi/v/pyrochain.svg)](https://pypi.org/project/pyrochain/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyrochain.svg)](https://pypi.org/project/pyrochain/)
[![Tests](https://github.com/irfanalidv/PyroChain/actions/workflows/tests.yml/badge.svg)](https://github.com/irfanalidv/PyroChain/actions/workflows/tests.yml)
[![PyPI Downloads](https://static.pepy.tech/badge/pyrochain)](https://pepy.tech/projects/pyrochain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/irfanalidv/PyroChain/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)](https://pytorch.org/)
[![LangChain](https://img.shields.io/badge/LangChain--core-0.3%2B-green)](https://python.langchain.com/)

</div>

---

PyroChain combines PyTorch and Hugging Face Transformers with LangChain message memory to turn
raw records (text, images, and structured fields) into feature tensors for machine-learning
models. A feature agent extracts features, a validation agent scores them and asks for refinement
when they fall below a threshold, and a lightweight LoRA adapter can be trained on your own text.

## 🎯 What problem does PyroChain solve?

**Traditional feature engineering is hard:**

- Manual feature extraction is time-consuming and error-prone
- Different data types require different approaches
- Domain expertise is needed to create meaningful features

**PyroChain gives you one pipeline for it:**

- One call turns a record (or a batch) with text, images, and numbers into feature tensors
- A validation agent scores every result (completeness, quality, relevance) and triggers refinement
- A LoRA adapter on top of a frozen open model can be fine-tuned on your domain text
- Plain PyTorch tensors out, so it plugs into existing ML pipelines

## 🚀 Key features

- **🤖 Feature and validation agents**: LangChain tools plus a ReAct-style reasoning trace, with
  conversation memory saved and restored as JSON
- **📊 Multimodal processing**: text embeddings from the base model's hidden states, image
  features (colour statistics, edges, size) via Pillow/OpenCV, fused by `concat`, `add`,
  `multiply`, or `attention`
- **⚡ Lightweight LoRA adapters**: the base model stays frozen; only low-rank layers train
  (about 12k trainable parameters at rank 8 on DialoGPT-small)
- **🛒 E-commerce helpers**: `EcommerceFeatureExtractor` and `ProductAnalyzer` (price, brand,
  review, inventory features; product similarity and clustering)
- **🖥️ CLI**: `pyrochain extract | train | predict | ecommerce | version`
- **📈 Metrics and visualisation**: `FeatureMetrics`, plus `FeatureVisualizer` (with the `viz` extra)

## 🛠️ Installation

```bash
pip install pyrochain               # core
pip install "pyrochain[viz]"        # + matplotlib / seaborn / pandas for FeatureVisualizer
```

Requires Python 3.10+. For a CPU-only PyTorch (smaller download), install it first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pyrochain
```

The first run downloads the default model (`microsoft/DialoGPT-small`, about 350 MB) and a
tokenizer from the Hugging Face Hub. Any Hub model with a `last_hidden_state` output works via
`PyroChainConfig(model_name=...)`.

## 🚀 Quick start

```python
from pyrochain import PyroChain, PyroChainConfig

chain = PyroChain(PyroChainConfig(device="cpu"))  # default model: microsoft/DialoGPT-small

product = {
    "title": "Wireless Bluetooth Headphones",
    "description": "Noise cancellation and 30-hour battery life.",
    "price": 199.99,
    "category": "electronics",
}
result = chain.extract_features(product, "Extract features for product recommendation")

features = result["features"][0]
print("modalities:", features["metadata"]["modalities"])
print("text features:", tuple(features["text_features"].shape))
print("validation score:", round(features["validation_score"], 2))
print(features["validation_feedback"])
```

Output:

```
modalities: ['text']
text features: (1, 768)
validation score: 1.0
Features passed validation with a score of 1.00
```

Pass a list of dicts to process a batch, and add a PIL image under `image` for multimodal
features. This script is in [`examples/quickstart.py`](https://github.com/irfanalidv/PyroChain/blob/main/examples/quickstart.py).

### Fine-tune the LoRA adapter

```python
data = [{"text": "Great headphones with long battery life."},
        {"text": "Comfortable organic cotton t-shirt."}] * 4
out = chain.train_adapter(data, "product language modelling", epochs=3, learning_rate=1e-3)
print([round(h["train_loss"], 2) for h in out["training_history"]])  # loss per epoch
chain.save_model("my_model")   # adapter weights + agent memory
chain.load_model("my_model")
```

On DialoGPT-small this run took the training loss from 9.08 to 6.41 over the 3 epochs.

### Command line

```bash
echo '{"title": "Organic Cotton T-Shirt", "price": 29.99}' > item.json
pyrochain extract --input-file item.json --task "product features" --output-file out.json --device cpu
pyrochain train --training-file train.json --task "domain LM" --epochs 3 --output-dir my_model
pyrochain predict --model-dir my_model --input-file item.json --task "product features"
pyrochain version
```

Every command accepts `--model` and `--device`. Tensors are written to JSON as nested lists.

## 💡 Use cases

**E-commerce and retail:** product recommendation features, product similarity and clustering,
price and inventory features, review sentiment signals.

**Content and media:** text classification and tagging features, image content descriptors,
content recommendation.

**Business intelligence:** feature tables for customer behaviour, trend, or risk models.

## 🏗️ How it works

1. **Data ingestion**: a dict (or list of dicts) with text fields, images, and scalars
2. **Processing**: `MultimodalProcessor` embeds text with the base model and extracts image features
3. **Feature agent**: runs its extraction tools and records a reasoning trace in memory
4. **Validation agent**: scores completeness, quality, and relevance against `validation_threshold`;
   below it, the feature chain runs again with the feedback
5. **Output**: `{"features": [...], "metadata": {...}}` with PyTorch tensors

## ⚙️ Configuration

`PyroChainConfig` fields (all optional):

| Field | Default | Meaning |
| --- | --- | --- |
| `model_name` | `"microsoft/DialoGPT-small"` | Hugging Face model used for embeddings and the adapter |
| `adapter_rank` | `16` | LoRA rank |
| `max_length` | `512` | Max tokens per text |
| `device` | `"auto"` | `auto` (CUDA, then MPS, then CPU), `cpu`, `cuda`, `mps` |
| `memory_type` | `"conversation_buffer"` | Agent memory type |
| `enable_validation` | `True` | Run the validation agent |
| `validation_threshold` | `0.8` | Minimum score before refinement is triggered |

No API keys are needed: everything runs locally on open models. Standard Hugging Face variables
such as `HF_HOME` and `HF_HUB_OFFLINE` apply.

## 📚 API reference

- **`PyroChain(config)`**: `extract_features(data, task_description, validate=True)`,
  `train_adapter(training_data, task_description, epochs, learning_rate)`, `save_model(path)`,
  `load_model(path)`
- **`PyroChainConfig`**: configuration dataclass (table above)
- **`LightweightAdapter` / `LoRAAdapter`**: `get_embeddings(text)`, `encode_text(text)`,
  `train(...)`, `save(path)`, `load(path)`, `get_trainable_parameters()`
- **`MultimodalProcessor`**: `process(record)` returns text, image, and fused features
- **`FeatureAgent` / `ValidationAgent`** and **`FeatureExtractionChain` / `ValidationChain`**
- **`pyrochain.utils`**: `EcommerceFeatureExtractor`, `ProductAnalyzer`, `ProductData`,
  `FeatureMetrics`, `FeatureVisualizer` (viz extra)

The built-in agents run their tools deterministically and do not call a generative LLM; the ReAct
prompt templates are exposed (`agent.agent`) if you want to drive them with your own LangChain model.

## 🤝 Contributing

Contributions are welcome! See the [Contributing Guide](https://github.com/irfanalidv/PyroChain/blob/main/CONTRIBUTING.md)
and [Code of Conduct](https://github.com/irfanalidv/PyroChain/blob/main/CODE_OF_CONDUCT.md).

```bash
git clone https://github.com/irfanalidv/PyroChain.git && cd PyroChain
pip install -e ".[dev]"
ruff check . && black --check . && pytest   # tests use a tiny offline model, no downloads
```

Security issues: see [SECURITY.md](https://github.com/irfanalidv/PyroChain/blob/main/SECURITY.md).
Release notes: [CHANGELOG.md](https://github.com/irfanalidv/PyroChain/blob/main/CHANGELOG.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/irfanalidv/PyroChain/blob/main/LICENSE) file for details.

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) for deep learning capabilities
- [LangChain](https://python.langchain.com/) for messages, tools, and prompts
- [Hugging Face](https://huggingface.co/) for transformer models

## 📞 Support

- 📚 [Documentation](https://github.com/irfanalidv/PyroChain#readme)
- 🐛 [Report Issues](https://github.com/irfanalidv/PyroChain/issues)
- 📧 [Contact](https://github.com/irfanalidv)

---

**PyroChain** - Transform your data into intelligent features with AI agents. 🔥

_Built with ❤️ by [Irfan Ali](https://github.com/irfanalidv)_
