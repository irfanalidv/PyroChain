"""Offline test fixtures: a tiny randomly initialised GPT-2 and a local WordPiece tokenizer.

Nothing is downloaded from the Hugging Face Hub, so the suite runs fast on CPU in CI.
"""

import pytest
import torch
from transformers import BertTokenizer, GPT2Config, GPT2Model

WORDS = (
    "this is a test product great good bad terrible headphones wireless battery life shirt "
    "cotton organic price review movie text title description extract features for the and"
).split()

HIDDEN = 32


@pytest.fixture(scope="session")
def vocab_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("tok") / "vocab.txt"
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(set(WORDS))
    tokens += list("abcdefghijklmnopqrstuvwxyz0123456789") + [
        f"##{c}" for c in "abcdefghijklmnopqrstuvwxyz"
    ]
    path.write_text("\n".join(tokens))
    return str(path)


@pytest.fixture(autouse=True)
def tiny_models(monkeypatch, vocab_file):
    """Patch AutoTokenizer/AutoModel.from_pretrained everywhere PyroChain uses them."""
    tokenizer = BertTokenizer(vocab_file)

    def fake_tokenizer(*args, **kwargs):
        return tokenizer

    def fake_model(*args, **kwargs):
        torch.manual_seed(0)
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size, n_embd=HIDDEN, n_layer=2, n_head=2, n_positions=128
        )
        return GPT2Model(config).eval()  # from_pretrained returns eval-mode models

    import pyrochain.adapters.base as adapters_base
    import pyrochain.processors.text as processors_text

    monkeypatch.setattr(adapters_base.AutoTokenizer, "from_pretrained", fake_tokenizer)
    monkeypatch.setattr(adapters_base.AutoModel, "from_pretrained", fake_model)
    monkeypatch.setattr(processors_text.AutoTokenizer, "from_pretrained", fake_tokenizer)
    return {"tokenizer": tokenizer, "hidden": HIDDEN}
