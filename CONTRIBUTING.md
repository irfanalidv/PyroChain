# Contributing to pyrochain

Thanks for helping improve `pyrochain`!

## Development setup

```bash
git clone https://github.com/irfanalidv/PyroChain.git
cd PyroChain
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check .          # lint
black --check .       # formatting
pytest                # tests
```

- Add or update tests for any behaviour change.
- Add a line to `CHANGELOG.md` under **Unreleased**.
- Keep pull requests focused: one fix or feature per PR.

## Reporting bugs and requesting features

Use the [issue templates](https://github.com/irfanalidv/PyroChain/issues/new/choose).
