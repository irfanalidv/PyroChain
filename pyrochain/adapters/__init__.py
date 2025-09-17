"""
Lightweight adapters for LLMs using PyTorch.

Implements efficient fine-tuning adapters for open-source LLMs like Gemma,
enabling task-specific adaptation without full model retraining.
"""

from .base import BaseAdapter
from .lora import LoRAAdapter
from .adapter import LightweightAdapter

__all__ = ["BaseAdapter", "LoRAAdapter", "LightweightAdapter"]
