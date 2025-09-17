"""
PyroChain: PyTorch + LangChain for Agentic Feature Engineering

A library that combines PyTorch's tensor operations with LangChain's memory-enabled chains
and Transformers for embedding open-source LLMs into data feature engineering tasks.
"""

__version__ = "0.1.0"
__author__ = "PyroChain Team"

from .core import PyroChain, PyroChainConfig
from .adapters import LightweightAdapter
from .agents import FeatureAgent, ValidationAgent
from .chains import FeatureExtractionChain, ValidationChain
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
]
