"""
PyroChain: PyTorch + LangChain for Agentic Feature Engineering

A library that combines PyTorch's tensor operations with LangChain's memory-enabled chains
and Transformers for embedding open-source LLMs into data feature engineering tasks.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyrochain")
except PackageNotFoundError:  # running from a source checkout without installation
    __version__ = "0.2.0"
__author__ = "PyroChain Team"

from .adapters import LightweightAdapter, LoRAAdapter
from .agents import FeatureAgent, ValidationAgent
from .chains import FeatureExtractionChain, ValidationChain
from .core import PyroChain, PyroChainConfig
from .processors import MultimodalProcessor

__all__ = [
    "PyroChain",
    "PyroChainConfig",
    "LightweightAdapter",
    "FeatureAgent",
    "ValidationAgent",
    "FeatureExtractionChain",
    "ValidationChain",
    "MultimodalProcessor",
    "LoRAAdapter",
    "__version__",
]
