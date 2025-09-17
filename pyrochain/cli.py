"""
Command-line interface for PyroChain.
"""

import click
import torch
import json
from pathlib import Path
from typing import Optional

from .core import PyroChain, PyroChainConfig
from .utils.ecommerce import EcommerceFeatureExtractor, ProductData


@click.group()
def main():
    """PyroChain: PyTorch + LangChain for Agentic Feature Engineering"""
    pass


@main.command()
@click.option('--model', default='google/gemma-2b', help='Model name to use')
@click.option('--device', default='auto', help='Device to use (auto, cpu, cuda, mps)')
@click.option('--adapter-rank', default=16, help='LoRA adapter rank')
@click.option('--max-length', default=512, help='Maximum sequence length')
def init(model: str, device: str, adapter_rank: int, max_length: int):
    """Initialize PyroChain with configuration."""
    config = PyroChainConfig(
        model_name=model,
        device=device,
        adapter_rank=adapter_rank,
        max_length=max_length
    )
    
    pyrochain = PyroChain(config)
    
    click.echo(f"PyroChain initialized with model: {model}")
    click.echo(f"Device: {pyrochain.device}")
    click.echo(f"Adapter rank: {adapter_rank}")
    click.echo(f"Max length: {max_length}")


@main.command()
@click.option('--input-file', required=True, help='Input data file (JSON)')
@click.option('--task', required=True, help='Feature extraction task description')
@click.option('--output-file', help='Output file for results')
@click.option('--validate/--no-validate', default=True, help='Enable validation')
def extract(input_file: str, task: str, output_file: Optional[str], validate: bool):
    """Extract features from input data."""
    # Load input data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Initialize PyroChain
    config = PyroChainConfig()
    pyrochain = PyroChain(config)
    
    # Extract features
    click.echo("Extracting features...")
    results = pyrochain.extract_features(
        data=data,
        task_description=task,
        validate=validate
    )
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        click.echo(f"Results saved to {output_file}")
    else:
        click.echo(json.dumps(results, indent=2, default=str))


@main.command()
@click.option('--training-file', required=True, help='Training data file (JSON)')
@click.option('--task', required=True, help='Training task description')
@click.option('--epochs', default=3, help='Number of training epochs')
@click.option('--learning-rate', default=1e-4, help='Learning rate')
@click.option('--output-dir', help='Output directory for trained model')
def train(training_file: str, task: str, epochs: int, learning_rate: float, output_dir: Optional[str]):
    """Train PyroChain adapter on custom data."""
    # Load training data
    with open(training_file, 'r') as f:
        training_data = json.load(f)
    
    # Initialize PyroChain
    config = PyroChainConfig()
    pyrochain = PyroChain(config)
    
    # Train adapter
    click.echo("Training adapter...")
    training_results = pyrochain.train_adapter(
        training_data=training_data,
        task_description=task,
        epochs=epochs,
        learning_rate=learning_rate
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
@click.option('--model-dir', required=True, help='Directory containing saved model')
@click.option('--input-file', required=True, help='Input data file (JSON)')
@click.option('--task', required=True, help='Feature extraction task description')
@click.option('--output-file', help='Output file for results')
def predict(model_dir: str, input_file: str, task: str, output_file: Optional[str]):
    """Make predictions using a trained model."""
    # Load input data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Initialize PyroChain
    config = PyroChainConfig()
    pyrochain = PyroChain(config)
    
    # Load trained model
    pyrochain.load_model(model_dir)
    
    # Make predictions
    click.echo("Making predictions...")
    results = pyrochain.extract_features(
        data=data,
        task_description=task,
        validate=True
    )
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        click.echo(f"Results saved to {output_file}")
    else:
        click.echo(json.dumps(results, indent=2, default=str))


@main.command()
@click.option('--product-file', required=True, help='Product data file (JSON)')
@click.option('--task', default='recommendation', help='E-commerce task type')
@click.option('--output-file', help='Output file for results')
def ecommerce(product_file: str, task: str, output_file: Optional[str]):
    """Extract features for e-commerce data."""
    # Load product data
    with open(product_file, 'r') as f:
        product_data = json.load(f)
    
    # Initialize PyroChain
    config = PyroChainConfig()
    pyrochain = PyroChain(config)
    
    # Initialize e-commerce feature extractor
    ecommerce_extractor = EcommerceFeatureExtractor(pyrochain)
    
    # Convert to ProductData object
    product = ProductData(**product_data)
    
    # Extract features
    click.echo("Extracting e-commerce features...")
    features = ecommerce_extractor.extract_product_features(product, task)
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(features, f, indent=2, default=str)
        click.echo(f"Results saved to {output_file}")
    else:
        click.echo(json.dumps(features, indent=2, default=str))


@main.command()
def version():
    """Show PyroChain version."""
    from . import __version__
    click.echo(f"PyroChain version {__version__}")


if __name__ == '__main__':
    main()
