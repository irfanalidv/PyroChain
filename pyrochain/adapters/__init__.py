"""
Lightweight adapters for LLMs using PyTorch.

Implements efficient fine-tuning adapters for open-source LLMs like Gemma,
enabling task-specific adaptation without full model retraining.
"""

from .adapter import LightweightAdapter
from .base import BaseAdapter
from .lora import LoRAAdapter

__all__ = ["BaseAdapter", "LoRAAdapter", "LightweightAdapter"]
