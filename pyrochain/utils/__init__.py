"""
Utility functions and helpers for PyroChain.
"""

from .ecommerce import EcommerceFeatureExtractor, ProductAnalyzer, ProductData
from .metrics import FeatureMetrics

__all__ = [
    "EcommerceFeatureExtractor",
    "ProductAnalyzer",
    "ProductData",
    "FeatureVisualizer",
    "FeatureMetrics",
]


def __getattr__(name):
    # FeatureVisualizer needs matplotlib/seaborn/pandas (``pip install "pyrochain[viz]"``),
    # so it is imported lazily.
    if name == "FeatureVisualizer":
        try:
            from .visualization import FeatureVisualizer
        except ImportError as exc:
            raise ImportError(
                "FeatureVisualizer requires the 'viz' extra: pip install \"pyrochain[viz]\""
            ) from exc
        return FeatureVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
