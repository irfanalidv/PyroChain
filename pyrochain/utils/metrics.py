"""
Metrics utilities for PyroChain.

Provides evaluation metrics for feature extraction,
validation, and model performance.
"""

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    silhouette_score,
)

ArrayLike = torch.Tensor | np.ndarray
"""A feature matrix as a PyTorch tensor or NumPy array."""


def as_numpy(values: ArrayLike) -> np.ndarray:
    """Return ``values`` as a NumPy array (tensors are detached and moved to CPU)."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


class FeatureMetrics:
    """Metrics for evaluating PyroChain features and models."""

    def __init__(self):
        """Initialize the metrics calculator."""

    def calculate_feature_quality_metrics(
        self, features: ArrayLike, ground_truth: ArrayLike | None = None
    ) -> dict[str, float]:
        """Calculate quality metrics for extracted features."""
        features_np = as_numpy(features)

        metrics = {}

        # Basic statistics
        metrics["mean"] = float(np.mean(features_np))
        metrics["std"] = float(np.std(features_np))
        metrics["min"] = float(np.min(features_np))
        metrics["max"] = float(np.max(features_np))
        metrics["range"] = float(np.max(features_np) - np.min(features_np))

        # Feature diversity
        metrics["unique_ratio"] = float(len(np.unique(features_np)) / features_np.size)
        metrics["zero_ratio"] = float(np.sum(features_np == 0) / features_np.size)

        # Feature distribution
        metrics["skewness"] = float(self._calculate_skewness(features_np))
        metrics["kurtosis"] = float(self._calculate_kurtosis(features_np))

        # Feature stability (if ground truth available)
        if ground_truth is not None:
            gt_np = as_numpy(ground_truth)

            # Ensure same shape
            if features_np.shape != gt_np.shape:
                # Reshape to match
                min_size = min(features_np.size, gt_np.size)
                features_flat = features_np.flatten()[:min_size]
                ground_truth_flat = gt_np.flatten()[:min_size]
            else:
                features_flat = features_np.flatten()
                ground_truth_flat = gt_np.flatten()

            # Correlation
            correlation = np.corrcoef(features_flat, ground_truth_flat)[0, 1]
            metrics["correlation"] = float(correlation) if not np.isnan(correlation) else 0.0

            # Mean squared error
            mse = mean_squared_error(ground_truth_flat, features_flat)
            metrics["mse"] = float(mse)

            # Mean absolute error
            mae = mean_absolute_error(ground_truth_flat, features_flat)
            metrics["mae"] = float(mae)

        return metrics

    def calculate_similarity_metrics(
        self, features1: ArrayLike, features2: ArrayLike
    ) -> dict[str, float]:
        """Calculate similarity metrics between two feature sets."""
        features1_np = as_numpy(features1)
        features2_np = as_numpy(features2)

        # Ensure same shape
        if features1_np.shape != features2_np.shape:
            # Reshape to match
            min_size = min(features1_np.size, features2_np.size)
            features1_np = features1_np.flatten()[:min_size]
            features2_np = features2_np.flatten()[:min_size]
        else:
            features1_np = features1_np.flatten()
            features2_np = features2_np.flatten()

        metrics = {}

        # Cosine similarity
        dot_product = np.dot(features1_np, features2_np)
        norm1 = np.linalg.norm(features1_np)
        norm2 = np.linalg.norm(features2_np)

        if norm1 == 0 or norm2 == 0:
            metrics["cosine_similarity"] = 0.0
        else:
            metrics["cosine_similarity"] = float(dot_product / (norm1 * norm2))

        # Euclidean distance
        euclidean_dist = np.linalg.norm(features1_np - features2_np)
        metrics["euclidean_distance"] = float(euclidean_dist)

        # Manhattan distance
        manhattan_dist = np.sum(np.abs(features1_np - features2_np))
        metrics["manhattan_distance"] = float(manhattan_dist)

        # Pearson correlation
        correlation = np.corrcoef(features1_np, features2_np)[0, 1]
        metrics["pearson_correlation"] = float(correlation) if not np.isnan(correlation) else 0.0

        # Jaccard similarity (for binary features)
        if np.all((features1_np == 0) | (features1_np == 1)) and np.all(
            (features2_np == 0) | (features2_np == 1)
        ):
            # Cast to bool: "&" / "|" are undefined for float arrays (TypeError before)
            b1, b2 = features1_np.astype(bool), features2_np.astype(bool)
            intersection = np.sum(b1 & b2)
            union = np.sum(b1 | b2)
            metrics["jaccard_similarity"] = float(intersection / union) if union > 0 else 0.0

        return metrics

    def calculate_validation_metrics(
        self, validation_results: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Calculate metrics for validation results."""
        if not validation_results:
            return {}

        scores = [r.get("score", 0) for r in validation_results]
        is_valid = [r.get("is_valid", False) for r in validation_results]

        metrics = {}

        # Basic statistics
        metrics["mean_score"] = float(np.mean(scores))
        metrics["std_score"] = float(np.std(scores))
        metrics["min_score"] = float(np.min(scores))
        metrics["max_score"] = float(np.max(scores))

        # Validation rate
        valid_count = sum(is_valid)
        total_count = len(validation_results)
        metrics["validation_rate"] = float(valid_count / total_count) if total_count > 0 else 0.0

        # Score distribution
        metrics["score_above_threshold"] = float(np.sum(np.array(scores) > 0.8) / total_count)
        metrics["score_below_threshold"] = float(np.sum(np.array(scores) < 0.5) / total_count)

        return metrics

    def calculate_clustering_metrics(
        self, features: ArrayLike, cluster_labels: list[int]
    ) -> dict[str, float]:
        """Calculate metrics for clustering results."""
        features_np = as_numpy(features)

        if features_np.ndim > 2:
            features_np = features_np.reshape(features_np.shape[0], -1)

        metrics = {}

        # Silhouette score
        if len(set(cluster_labels)) > 1:
            silhouette = silhouette_score(features_np, cluster_labels)
            metrics["silhouette_score"] = float(silhouette)
        else:
            metrics["silhouette_score"] = 0.0

        # Number of clusters
        metrics["num_clusters"] = float(len(set(cluster_labels)))

        # Cluster size distribution
        cluster_sizes = [cluster_labels.count(i) for i in set(cluster_labels)]
        metrics["mean_cluster_size"] = float(np.mean(cluster_sizes))
        metrics["std_cluster_size"] = float(np.std(cluster_sizes))
        metrics["min_cluster_size"] = float(np.min(cluster_sizes))
        metrics["max_cluster_size"] = float(np.max(cluster_sizes))

        # Cluster balance
        if len(cluster_sizes) > 1:
            balance = np.min(cluster_sizes) / np.max(cluster_sizes)
            metrics["cluster_balance"] = float(balance)
        else:
            metrics["cluster_balance"] = 1.0

        return metrics

    def calculate_training_metrics(
        self, training_history: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Calculate metrics for training history."""
        if not training_history:
            return {}

        train_losses = [h.get("train_loss", 0) for h in training_history]
        val_losses = [h.get("val_loss", 0) for h in training_history if "val_loss" in h]

        metrics = {}

        # Training metrics
        metrics["final_train_loss"] = float(train_losses[-1]) if train_losses else 0.0
        metrics["min_train_loss"] = float(np.min(train_losses)) if train_losses else 0.0
        metrics["train_loss_improvement"] = (
            float(train_losses[0] - train_losses[-1]) if len(train_losses) > 1 else 0.0
        )

        # Validation metrics
        if val_losses:
            metrics["final_val_loss"] = float(val_losses[-1])
            metrics["min_val_loss"] = float(np.min(val_losses))
            metrics["val_loss_improvement"] = (
                float(val_losses[0] - val_losses[-1]) if len(val_losses) > 1 else 0.0
            )

            # Overfitting detection
            if len(train_losses) == len(val_losses):
                overfitting = np.mean(val_losses[-3:]) - np.mean(train_losses[-3:])
                metrics["overfitting_indicator"] = float(overfitting)

        # Convergence metrics
        if len(train_losses) > 5:
            recent_losses = train_losses[-5:]
            convergence = np.std(recent_losses)
            metrics["convergence_std"] = float(convergence)

        return metrics

    def calculate_ecommerce_metrics(
        self, product_features: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Calculate metrics specific to e-commerce features."""
        if not product_features:
            return {}

        metrics = {}

        # Price analysis
        prices = []
        for features in product_features:
            if "ecommerce_features" in features:
                price = features["ecommerce_features"].get("price_features", {}).get("price", 0)
                if price > 0:
                    prices.append(price)

        if prices:
            metrics["mean_price"] = float(np.mean(prices))
            metrics["std_price"] = float(np.std(prices))
            metrics["price_range"] = float(np.max(prices) - np.min(prices))

        # Category distribution
        categories = []
        for features in product_features:
            if "ecommerce_features" in features:
                category = (
                    features["ecommerce_features"].get("category_features", {}).get("category", "")
                )
                if category:
                    categories.append(category)

        if categories:
            unique_categories = len(set(categories))
            total_products = len(categories)
            metrics["category_diversity"] = float(unique_categories / total_products)
            metrics["num_categories"] = float(unique_categories)

        # Review analysis
        review_scores = []
        for features in product_features:
            if "ecommerce_features" in features:
                avg_rating = (
                    features["ecommerce_features"].get("review_features", {}).get("avg_rating", 0)
                )
                if avg_rating > 0:
                    review_scores.append(avg_rating)

        if review_scores:
            metrics["mean_rating"] = float(np.mean(review_scores))
            metrics["std_rating"] = float(np.std(review_scores))
            metrics["high_rated_products"] = float(
                np.sum(np.array(review_scores) >= 4.0) / len(review_scores)
            )

        return metrics

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4)) - 3

    def generate_comprehensive_report(
        self,
        features: dict[str, Any],
        validation_results: list[dict[str, Any]] | None = None,
        training_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate a comprehensive metrics report."""
        report = {}

        # Feature quality metrics
        if "raw_features" in features and isinstance(features["raw_features"], torch.Tensor):
            report["feature_quality"] = self.calculate_feature_quality_metrics(
                features["raw_features"]
            )

        # Validation metrics
        if validation_results:
            report["validation_metrics"] = self.calculate_validation_metrics(validation_results)

        # Training metrics
        if training_history:
            report["training_metrics"] = self.calculate_training_metrics(training_history)

        # E-commerce metrics
        if isinstance(features, list):
            report["ecommerce_metrics"] = self.calculate_ecommerce_metrics(features)
        elif "ecommerce_features" in features:
            report["ecommerce_metrics"] = self.calculate_ecommerce_metrics([features])

        return report
