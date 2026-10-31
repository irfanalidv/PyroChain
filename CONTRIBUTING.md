# Contributing to pyrochain

Thanks for helping improve `pyrochain`!

## Development setup

```bash
git clone https://github.com/irfanalidv/PyroChain.git
cd PyroChain
uv venv && source .venv/bin/activate          # or: python -m venv .venv
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev,viz]"                  # or: pip install -e ".[dev,viz]"
pre-commit install                              # optional: ruff + hygiene hooks on commit
```

## Before opening a pull request

```bash
ruff check .                 # lint
ruff format --check .        # formatting
mypy pyrochain               # types
pytest --cov=pyrochain       # tests (tiny offline model, no downloads)
```

- Add or update tests for any behaviour change.
- Add a line to `CHANGELOG.md` under **Unreleased**.
- Keep pull requests focused: one fix or feature per PR.

## Reporting bugs and requesting features

Use the [issue templates](https://github.com/irfanalidv/PyroChain/issues/new/choose).
