"""
Command-line interface for PyroChain.
"""

import json
from pathlib import Path

import click
import torch

from .core import PyroChain, PyroChainConfig
from .utils.ecommerce import EcommerceFeatureExtractor, ProductData

DEFAULT_MODEL = PyroChainConfig.model_name

model_option = click.option(
    "--model", default=DEFAULT_MODEL, show_default=True, help="Hugging Face model name"
)
device_option = click.option(
    "--device", default="auto", show_default=True, help="Device: auto, cpu, cuda, mps"
)


def _to_jsonable(obj):
    """Convert tensors / numpy values in results to plain JSON types."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except (TypeError, ValueError):
            pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _load_json(path: str):
    """Read a JSON input file with a clear error message."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise click.BadParameter(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise click.BadParameter(f"{path} is not valid JSON: {exc}") from exc


def _emit(results, output_file):
    payload = json.dumps(_to_jsonable(results), indent=2)
    if output_file:
        Path(output_file).write_text(payload)
        click.echo(f"Results saved to {output_file}")
    else:
        click.echo(payload)


@click.group()
def main():
    """PyroChain: PyTorch + LangChain for Agentic Feature Engineering"""


@main.command()
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Hugging Face model name")
@click.option("--device", default="auto", help="Device to use (auto, cpu, cuda, mps)")
@click.option("--adapter-rank", default=16, help="LoRA adapter rank")
@click.option("--max-length", default=512, help="Maximum sequence length")
def init(model: str, device: str, adapter_rank: int, max_length: int):
    """Initialize PyroChain with configuration."""
    config = PyroChainConfig(
        model_name=model, device=device, adapter_rank=adapter_rank, max_length=max_length
    )

    pyrochain = PyroChain(config)

    click.echo(f"PyroChain initialized with model: {model}")
    click.echo(f"Device: {pyrochain.device}")
    click.echo(f"Adapter rank: {adapter_rank}")
    click.echo(f"Max length: {max_length}")


@main.command()
@click.option("--input-file", required=True, help="Input data file (JSON)")
@click.option("--task", required=True, help="Feature extraction task description")
@click.option("--output-file", help="Output file for results")
@click.option("--validate/--no-validate", default=True, help="Enable validation")
@model_option
@device_option
def extract(
    input_file: str, task: str, output_file: str | None, validate: bool, model: str, device: str
):
    """Extract features from input data."""
    # Load input data
    data = _load_json(input_file)

    # Initialize PyroChain
    config = PyroChainConfig(model_name=model, device=device)
    pyrochain = PyroChain(config)

    # Extract features
    click.echo("Extracting features...")
    results = pyrochain.extract_features(data=data, task_description=task, validate=validate)

    _emit(results, output_file)


@main.command()
@click.option("--training-file", required=True, help="Training data file (JSON)")
@click.option("--task", required=True, help="Training task description")
@click.option("--epochs", default=3, help="Number of training epochs")
@click.option("--learning-rate", default=1e-4, help="Learning rate")
@click.option("--output-dir", help="Output directory for trained model")
@model_option
@device_option
def train(
    training_file: str,
    task: str,
    epochs: int,
    learning_rate: float,
    output_dir: str | None,
    model: str,
    device: str,
):
    """Train PyroChain adapter on custom data."""
    # Load training data
    training_data = _load_json(training_file)

    # Initialize PyroChain
    config = PyroChainConfig(model_name=model, device=device)
    pyrochain = PyroChain(config)

    # Train adapter
    click.echo("Training adapter...")
    training_results = pyrochain.train_adapter(
        training_data=training_data,
        task_description=task,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    # Save model
    if output_dir:
        pyrochain.save_model(output_dir)
        click.echo(f"Model saved to {output_dir}")

    # Display results
    click.echo("Training completed!")
    click.echo(f"Final loss: {training_results['training_history'][-1]['train_loss']:.4f}")
    click.echo(f"Total parameters: {training_results['total_parameters']}")


@main.command()
@click.option("--model-dir", required=True, help="Directory containing saved model")
@click.option("--input-file", required=True, help="Input data file (JSON)")
@click.option("--task", required=True, help="Feature extraction task description")
@click.option("--output-file", help="Output file for results")
@model_option
@device_option
def predict(
    model_dir: str, input_file: str, task: str, output_file: str | None, model: str, device: str
):
    """Make predictions using a trained model."""
    # Load input data
    data = _load_json(input_file)

    # Initialize PyroChain
    config = PyroChainConfig(model_name=model, device=device)
    pyrochain = PyroChain(config)

    # Load trained model
    pyrochain.load_model(model_dir)

    # Make predictions
    click.echo("Making predictions...")
    results = pyrochain.extract_features(data=data, task_description=task, validate=True)

    _emit(results, output_file)


@main.command()
@click.option("--product-file", required=True, help="Product data file (JSON)")
@click.option("--task", default="recommendation", help="E-commerce task type")
@click.option("--output-file", help="Output file for results")
@model_option
@device_option
def ecommerce(product_file: str, task: str, output_file: str | None, model: str, device: str):
    """Extract features for e-commerce data."""
    # Load product data
    product_data = _load_json(product_file)

    # Initialize PyroChain
    config = PyroChainConfig(model_name=model, device=device)
    pyrochain = PyroChain(config)

    # Initialize e-commerce feature extractor
    ecommerce_extractor = EcommerceFeatureExtractor(pyrochain)

    # Convert to ProductData object
    product = ProductData(**product_data)

    # Extract features
    click.echo("Extracting e-commerce features...")
    features = ecommerce_extractor.extract_product_features(product, task)

    _emit(features, output_file)


@main.command()
def version():
    """Show PyroChain version."""
    from . import __version__

    click.echo(f"PyroChain version {__version__}")


if __name__ == "__main__":
    main()
