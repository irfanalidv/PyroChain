"""
Metrics utilities for PyroChain.

Provides evaluation metrics for feature extraction,
validation, and model performance.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score,
    adjusted_rand_score,
)


class FeatureMetrics:
    """Metrics for evaluating PyroChain features and models."""

    def __init__(self):
        """Initialize the metrics calculator."""
        pass

    def calculate_feature_quality_metrics(
        self, features: torch.Tensor, ground_truth: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Calculate quality metrics for extracted features."""
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        metrics = {}

        # Basic statistics
        metrics["mean"] = float(np.mean(features))
        metrics["std"] = float(np.std(features))
        metrics["min"] = float(np.min(features))
        metrics["max"] = float(np.max(features))
        metrics["range"] = float(np.max(features) - np.min(features))

        # Feature diversity
        metrics["unique_ratio"] = float(len(np.unique(features)) / features.size)
        metrics["zero_ratio"] = float(np.sum(features == 0) / features.size)

        # Feature distribution
        metrics["skewness"] = float(self._calculate_skewness(features))
        metrics["kurtosis"] = float(self._calculate_kurtosis(features))

        # Feature stability (if ground truth available)
        if ground_truth is not None:
            if isinstance(ground_truth, torch.Tensor):
                ground_truth = ground_truth.detach().cpu().numpy()

            # Ensure same shape
            if features.shape != ground_truth.shape:
                # Reshape to match
                min_size = min(features.size, ground_truth.size)
                features_flat = features.flatten()[:min_size]
                ground_truth_flat = ground_truth.flatten()[:min_size]
            else:
                features_flat = features.flatten()
                ground_truth_flat = ground_truth.flatten()

            # Correlation
            correlation = np.corrcoef(features_flat, ground_truth_flat)[0, 1]
            metrics["correlation"] = (
                float(correlation) if not np.isnan(correlation) else 0.0
            )

            # Mean squared error
            mse = mean_squared_error(ground_truth_flat, features_flat)
            metrics["mse"] = float(mse)

            # Mean absolute error
            mae = mean_absolute_error(ground_truth_flat, features_flat)
            metrics["mae"] = float(mae)

        return metrics

    def calculate_similarity_metrics(
        self, features1: torch.Tensor, features2: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate similarity metrics between two feature sets."""
        if isinstance(features1, torch.Tensor):
            features1 = features1.detach().cpu().numpy()
        if isinstance(features2, torch.Tensor):
            features2 = features2.detach().cpu().numpy()

        # Ensure same shape
        if features1.shape != features2.shape:
            # Reshape to match
            min_size = min(features1.size, features2.size)
            features1 = features1.flatten()[:min_size]
            features2 = features2.flatten()[:min_size]
        else:
            features1 = features1.flatten()
            features2 = features2.flatten()

        metrics = {}

        # Cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)

        if norm1 == 0 or norm2 == 0:
            metrics["cosine_similarity"] = 0.0
        else:
            metrics["cosine_similarity"] = float(dot_product / (norm1 * norm2))

        # Euclidean distance
        euclidean_dist = np.linalg.norm(features1 - features2)
        metrics["euclidean_distance"] = float(euclidean_dist)

        # Manhattan distance
        manhattan_dist = np.sum(np.abs(features1 - features2))
        metrics["manhattan_distance"] = float(manhattan_dist)

        # Pearson correlation
        correlation = np.corrcoef(features1, features2)[0, 1]
        metrics["pearson_correlation"] = (
            float(correlation) if not np.isnan(correlation) else 0.0
        )

        # Jaccard similarity (for binary features)
        if np.all((features1 == 0) | (features1 == 1)) and np.all(
            (features2 == 0) | (features2 == 1)
        ):
            intersection = np.sum(features1 & features2)
            union = np.sum(features1 | features2)
            metrics["jaccard_similarity"] = (
                float(intersection / union) if union > 0 else 0.0
            )

        return metrics

    def calculate_validation_metrics(
        self, validation_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
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
        metrics["validation_rate"] = (
            float(valid_count / total_count) if total_count > 0 else 0.0
        )

        # Score distribution
        metrics["score_above_threshold"] = float(
            np.sum(np.array(scores) > 0.8) / total_count
        )
        metrics["score_below_threshold"] = float(
            np.sum(np.array(scores) < 0.5) / total_count
        )

        return metrics

    def calculate_clustering_metrics(
        self, features: torch.Tensor, cluster_labels: List[int]
    ) -> Dict[str, float]:
        """Calculate metrics for clustering results."""
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)

        metrics = {}

        # Silhouette score
        if len(set(cluster_labels)) > 1:
            silhouette = silhouette_score(features, cluster_labels)
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
        self, training_history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
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
        self, product_features: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate metrics specific to e-commerce features."""
        if not product_features:
            return {}

        metrics = {}

        # Price analysis
        prices = []
        for features in product_features:
            if "ecommerce_features" in features:
                price = (
                    features["ecommerce_features"]
                    .get("price_features", {})
                    .get("price", 0)
                )
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
                    features["ecommerce_features"]
                    .get("category_features", {})
                    .get("category", "")
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
                    features["ecommerce_features"]
                    .get("review_features", {})
                    .get("avg_rating", 0)
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
        features: Dict[str, Any],
        validation_results: Optional[List[Dict[str, Any]]] = None,
        training_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Generate a comprehensive metrics report."""
        report = {}

        # Feature quality metrics
        if "raw_features" in features and isinstance(
            features["raw_features"], torch.Tensor
        ):
            report["feature_quality"] = self.calculate_feature_quality_metrics(
                features["raw_features"]
            )

        # Validation metrics
        if validation_results:
            report["validation_metrics"] = self.calculate_validation_metrics(
                validation_results
            )

        # Training metrics
        if training_history:
            report["training_metrics"] = self.calculate_training_metrics(
                training_history
            )

        # E-commerce metrics
        if isinstance(features, list):
            report["ecommerce_metrics"] = self.calculate_ecommerce_metrics(features)
        elif "ecommerce_features" in features:
            report["ecommerce_metrics"] = self.calculate_ecommerce_metrics([features])

        return report
