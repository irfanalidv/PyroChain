# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-10-31

### Fixed
- Text embeddings ignored the LoRA adapter (`LoRAAdapter` used the base model's pooled output),
  so `train_adapter()` never changed extracted features. Embeddings now come from the adapted
  hidden states; an untrained adapter still returns the base-model embedding.
- Validation credited every record with "text and image features present" (the keys always
  exist), so empty records passed. Scores now depend on detected modalities and non-zero
  features: text-only 0.91, text + image 1.0, empty records fail. Refined features are
  re-validated and marked `refined=True`; `ValidationChain` feedback now inspects the validated
  features instead of a key that never existed.
- `ProductAnalyzer` never computed text or image similarity and `cluster_products` always
  returned "No valid features found" (it read tensors from the wrong level of the result).
- `category_encoded` / `brand_encoded` used Python's per-process randomised `hash()`; they now
  use a stable CRC-32 bucket (`stable_bucket()`). Boolean attributes were encoded as numbers.
- `ImageProcessor.extract_colors` crashed on every call and on grayscale / RGBA images.
- `MultimodalProcessor.get_feature_summary` raised `KeyError`.
- `FeatureMetrics` Jaccard similarity raised `TypeError` on float arrays.
- Adapter training on a dataset too small for the validation split trained on nothing; empty
  or text-less data now raises `ValueError`. `LoRAAdapter.load()` silently ignored a missing
  weights file (now `FileNotFoundError`), and reloaded layers ran with dropout on.
- `FeatureVisualizer` showed (blocking) after saving and never closed figures.
- CLI input files that are missing or not JSON give a usage error instead of a traceback.

### Added
- `EcommerceFeatureExtractor.extract_tabular_features()`; `ProductData.images` are used (first
  PIL image or local file path).
- `LightweightAdapter.train(seed=...)` for reproducible splits (no global RNG use).
- Four examples with sample data: review sentiment classifier, catalog similarity and
  clustering, listing feature matrix, adapter fine-tuning.
- CodeQL workflow, pre-commit config, mypy and an 85% coverage gate in CI.

### Changed
- Training progress is logged (`logging`) instead of printed; progress bars hide on non-TTY.
- Formatting moved from black to `ruff format`; ruff rules broadened; mypy runs with
  `check_untyped_defs`. Test coverage 63% -> 89%.
- README rewritten: architecture diagram, design decisions, configuration and API tables,
  references. Removed `examples/main_example.py`, which did not use PyroChain.

## [0.2.0] - 2026-10-31

### Fixed
- `import pyrochain` failed on LangChain 1.x (`AgentExecutor`, `initialize_agent`, `Tool`,
  `langchain.memory`, `langchain.schema` were removed). PyroChain now depends only on
  `langchain-core` (messages, tools, prompts).
- `PyroChain()` could not be constructed even on older LangChain: the agents passed a
  non-LangChain placeholder LLM to `initialize_agent`, which rejects it. Agents now use a
  deterministic `SequentialToolExecutor` that runs their tools and records a ReAct-style trace,
  and a bounded `ConversationMemory`.
- Text features were random (`torch.randn` placeholder). They are now mean-pooled hidden states
  from the base model (via the adapter), with a deterministic hashed bag-of-tokens fallback.
- LoRA adapter: layer names containing `.` crashed `nn.ModuleDict`; adapters only target
  hidden-to-hidden `nn.Linear` modules and fall back to one residual layer on the final hidden
  states (GPT-2 uses `Conv1D`, so previously no layer was created and training had no parameters).
- Adapter training could not run: missing `train()`/`eval()`, double `backward()`, `.backward()`
  on a float, loss over hidden size instead of vocabulary, unpadded batches, and an f-string
  crash when there was no validation set. Training now computes a proper next-token loss through
  the tied embedding matrix and was verified to reduce loss on `microsoft/DialoGPT-small`.
- Already-processed data was processed a second time, and feature tensors were mistaken for
  images (shape errors); grayscale images are converted to RGB before normalisation.
- `if not tensor` checks in the refinement logic raised "Boolean value of Tensor is ambiguous".
- Validation metadata used CUDA events incorrectly (would crash on GPU); it now records a timestamp.
- Agent memory is saved as JSON (`memory.json`) instead of pickle (unsafe to load); LoRA weights
  load with `weights_only=True`.
- The CLI `init` default model was the gated `google/gemma-2b`; it now matches the library
  default (`microsoft/DialoGPT-small`).

### Added
- CLI: `--model` and `--device` on `extract`, `train`, `predict`, `ecommerce`; JSON output with
  tensors converted to lists.
- `examples/quickstart.py` that uses the library.
- `py.typed`, `__version__` from package metadata, `LoRAAdapter` exported at top level.
- 37 tests running fully offline on a tiny randomly initialised GPT-2 (adapters, training,
  end-to-end extraction, images, validation/refinement, memory persistence, e-commerce, CLI).
- GitHub Actions CI (ruff + black, pytest on Python 3.10 to 3.13 with CPU-only PyTorch).
- CHANGELOG, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, issue/PR templates, Dependabot, CITATION.cff.

### Changed
- Requires Python 3.10+; classifiers for 3.10 to 3.13.
- Packaging is `pyproject.toml`-only with a static version (removed `setup.py`,
  `requirements.txt`, `setuptools_scm`), PEP 639 `license = "MIT"`.
- Removed unused dependencies (`langchain`, `langchain-community`, `langchain-experimental`,
  `chromadb`, `faiss-cpu`, `sentence-transformers`, `redis`, `datasets`, `textblob`, `nltk`,
  `rich`, `python-dotenv`, `pydantic`). `matplotlib`/`seaborn`/`pandas` moved to the `viz` extra;
  `opencv-python` replaced by `opencv-python-headless`.
- Removed committed `__pycache__` files; code formatted with black and linted with ruff.
- README rewritten around code that was actually run; the previous configuration example used
  `PyroChainConfig` fields and methods that do not exist.

## [0.1.0] - 2025-09-17

- Initial release.
