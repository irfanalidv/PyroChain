"""
Fine-tune PyroChain's LoRA adapter on your own domain text, then save and reload it.

Shows:
- ``PyroChain.train_adapter``: only the low-rank layers train; the base model stays frozen.
- Per-epoch training and validation loss, and the number of trainable parameters.
- ``save_model`` / ``load_model`` round trip (adapter weights + agent memory), checked by
  comparing embeddings before and after reloading.

Data: ``examples/assets/support_corpus.txt`` (16 customer-support sentences written for this
example). Needs: ``pip install pyrochain``. No API key. The first run downloads
``microsoft/DialoGPT-small`` (about 350 MB). Weights are written to a temporary folder.

Run: python examples/support_adapter_finetune.py
"""

import tempfile
from pathlib import Path

import torch
from transformers.utils import logging as hf_logging

from pyrochain import PyroChain, PyroChainConfig

CORPUS = Path(__file__).parent / "assets" / "support_corpus.txt"


def main() -> None:
    hf_logging.set_verbosity_error()
    torch.manual_seed(0)
    lines = [line for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]

    chain = PyroChain(PyroChainConfig(device="cpu", adapter_rank=8))
    out = chain.adapter.train(
        [{"text": line} for line in lines],
        "customer-support language modelling",
        epochs=4,
        learning_rate=2e-3,
        validation_split=0.25,
        seed=0,
    )
    print(f"Trainable parameters: {out['total_parameters']:,} (base model frozen)")
    for h in out["training_history"]:
        print(
            f"  epoch {h['epoch']}: train loss {h['train_loss']:.2f}, val loss {h['val_loss']:.2f}"
        )

    probe = "Where is my refund?"
    before = chain.adapter.get_embeddings(probe)
    with tempfile.TemporaryDirectory() as tmp:
        chain.save_model(tmp)
        saved = sorted(p.relative_to(tmp).as_posix() for p in Path(tmp).rglob("*") if p.is_file())
        print("Saved files:", ", ".join(saved))

        fresh = PyroChain(PyroChainConfig(device="cpu", adapter_rank=8))
        fresh.load_model(tmp)
        after = fresh.adapter.get_embeddings(probe)
    print("Embeddings identical after reload:", torch.allclose(before, after, atol=1e-6))


if __name__ == "__main__":
    main()
