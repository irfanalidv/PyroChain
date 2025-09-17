"""
Visualization utilities for PyroChain.

Provides tools for visualizing extracted features,
similarity matrices, and clustering results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class FeatureVisualizer:
    """Visualization utilities for PyroChain features."""

    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """Initialize the visualizer."""
        self.figsize = figsize
        plt.style.use("seaborn-v0_8")

    def plot_feature_distribution(
        self,
        features: torch.Tensor,
        title: str = "Feature Distribution",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot distribution of feature values."""
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        plt.figure(figsize=self.figsize)

        if features.ndim > 1:
            features = features.flatten()

        plt.hist(features, bins=50, alpha=0.7, edgecolor="black")
        plt.title(title)
        plt.xlabel("Feature Value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_similarity_matrix(
        self,
        similarity_matrix: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Similarity Matrix",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot similarity matrix as heatmap."""
        plt.figure(figsize=self.figsize)

        sns.heatmap(
            similarity_matrix,
            annot=True,
            cmap="viridis",
            center=0,
            square=True,
            xticklabels=labels,
            yticklabels=labels,
        )

        plt.title(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_feature_embedding(
        self,
        features: torch.Tensor,
        labels: Optional[List[str]] = None,
        method: str = "pca",
        n_components: int = 2,
        title: str = "Feature Embedding",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot feature embeddings in 2D space."""
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)

        # Reduce dimensionality
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components, random_state=42)
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")

        embeddings = reducer.fit_transform(features)

        plt.figure(figsize=self.figsize)

        if labels is not None:
            unique_labels = list(set(labels))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = [l == label for l in labels]
                plt.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c=[colors[i]],
                    label=label,
                    alpha=0.7,
                )
            plt.legend()
        else:
            plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.7)

        plt.title(title)
        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_clustering_results(
        self,
        features: torch.Tensor,
        cluster_labels: List[int],
        title: str = "Clustering Results",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot clustering results."""
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)

        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        embeddings = pca.fit_transform(features)

        plt.figure(figsize=self.figsize)

        unique_clusters = list(set(cluster_labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))

        for i, cluster in enumerate(unique_clusters):
            mask = [c == cluster for c in cluster_labels]
            plt.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=[colors[i]],
                label=f"Cluster {cluster}",
                alpha=0.7,
            )

        plt.title(title)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_feature_importance(
        self,
        importance_scores: Dict[str, float],
        title: str = "Feature Importance",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot feature importance scores."""
        features = list(importance_scores.keys())
        scores = list(importance_scores.values())

        # Sort by importance
        sorted_indices = np.argsort(scores)[::-1]
        features = [features[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]

        plt.figure(figsize=self.figsize)

        bars = plt.bar(range(len(features)), scores, color="skyblue", edgecolor="navy")
        plt.title(title)
        plt.xlabel("Features")
        plt.ylabel("Importance Score")
        plt.xticks(range(len(features)), features, rotation=45, ha="right")
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_training_history(
        self,
        history: List[Dict[str, Any]],
        metrics: List[str] = ["train_loss", "val_loss"],
        title: str = "Training History",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot training history."""
        if not history:
            return

        epochs = [h["epoch"] for h in history]

        plt.figure(figsize=self.figsize)

        for metric in metrics:
            values = [h.get(metric, None) for h in history]
            valid_values = [(e, v) for e, v in zip(epochs, values) if v is not None]

            if valid_values:
                epochs_metric, values_metric = zip(*valid_values)
                plt.plot(epochs_metric, values_metric, label=metric, marker="o")

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_validation_metrics(
        self,
        validation_results: List[Dict[str, Any]],
        title: str = "Validation Metrics",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot validation metrics."""
        if not validation_results:
            return

        scores = [r.get("score", 0) for r in validation_results]
        is_valid = [r.get("is_valid", False) for r in validation_results]

        plt.figure(figsize=self.figsize)

        # Plot scores
        plt.subplot(2, 1, 1)
        plt.plot(scores, marker="o", color="blue")
        plt.title("Validation Scores")
        plt.ylabel("Score")
        plt.grid(True, alpha=0.3)

        # Plot validity
        plt.subplot(2, 1, 2)
        valid_counts = [int(v) for v in is_valid]
        plt.plot(valid_counts, marker="s", color="green")
        plt.title("Validation Results (1=Valid, 0=Invalid)")
        plt.xlabel("Sample Index")
        plt.ylabel("Valid")
        plt.grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def create_feature_report(
        self, features: Dict[str, Any], save_path: Optional[str] = None
    ) -> None:
        """Create a comprehensive feature report."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Feature distribution
        if "raw_features" in features and isinstance(
            features["raw_features"], torch.Tensor
        ):
            raw_features = features["raw_features"].detach().cpu().numpy()
            axes[0, 0].hist(raw_features.flatten(), bins=50, alpha=0.7)
            axes[0, 0].set_title("Raw Features Distribution")
            axes[0, 0].set_xlabel("Feature Value")
            axes[0, 0].set_ylabel("Frequency")

        # Modalities
        if "metadata" in features and "modalities" in features["metadata"]:
            modalities = features["metadata"]["modalities"]
            axes[0, 1].pie([1] * len(modalities), labels=modalities, autopct="%1.1f%%")
            axes[0, 1].set_title("Detected Modalities")

        # Feature importance (if available)
        if "ecommerce_features" in features:
            ecommerce_features = features["ecommerce_features"]
            feature_names = list(ecommerce_features.keys())
            feature_counts = [
                len(str(v)) if isinstance(v, (dict, list)) else 1
                for v in ecommerce_features.values()
            ]

            axes[1, 0].bar(feature_names, feature_counts)
            axes[1, 0].set_title("E-commerce Features")
            axes[1, 0].set_ylabel("Feature Count")
            axes[1, 0].tick_params(axis="x", rotation=45)

        # Summary statistics
        if "raw_features" in features and isinstance(
            features["raw_features"], torch.Tensor
        ):
            raw_features = features["raw_features"].detach().cpu().numpy()
            stats = {
                "Mean": np.mean(raw_features),
                "Std": np.std(raw_features),
                "Min": np.min(raw_features),
                "Max": np.max(raw_features),
            }

            axes[1, 1].bar(stats.keys(), stats.values())
            axes[1, 1].set_title("Feature Statistics")
            axes[1, 1].set_ylabel("Value")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
