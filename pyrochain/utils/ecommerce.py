"""
E-commerce specific utilities for PyroChain.

Provides specialized feature extraction and analysis tools
for e-commerce applications like product recommendation,
search optimization, and customer behavior analysis.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from PIL import Image
import json


@dataclass
class ProductData:
    """Data structure for e-commerce products."""
    product_id: str
    title: str
    description: str
    price: float
    category: str
    brand: str
    images: List[Union[str, Image.Image]]
    attributes: Dict[str, Any]
    reviews: List[Dict[str, Any]]
    inventory: int


class EcommerceFeatureExtractor:
    """
    Specialized feature extractor for e-commerce data.
    
    Extracts features optimized for e-commerce tasks like
    product recommendation, search ranking, and customer analysis.
    """
    
    def __init__(self, pyrochain_instance):
        """Initialize with a PyroChain instance."""
        self.pyrochain = pyrochain_instance
    
    def extract_product_features(
        self, 
        product_data: ProductData,
        task_type: str = "recommendation"
    ) -> Dict[str, Any]:
        """
        Extract features from product data.
        
        Args:
            product_data: Product data to process
            task_type: Type of e-commerce task
            
        Returns:
            Dictionary containing extracted features
        """
        # Prepare data for PyroChain
        input_data = self._prepare_product_data(product_data)
        
        # Create task-specific description
        task_description = self._create_task_description(task_type)
        
        # Extract features using PyroChain
        features = self.pyrochain.extract_features(
            data=input_data,
            task_description=task_description,
            validate=True
        )
        
        # Add e-commerce specific features
        ecommerce_features = self._extract_ecommerce_specific_features(product_data)
        
        # Combine features
        combined_features = {
            **features,
            "ecommerce_features": ecommerce_features,
            "product_metadata": {
                "product_id": product_data.product_id,
                "category": product_data.category,
                "brand": product_data.brand,
                "price": product_data.price,
                "task_type": task_type
            }
        }
        
        return combined_features
    
    def _prepare_product_data(self, product_data: ProductData) -> Dict[str, Any]:
        """Prepare product data for PyroChain processing."""
        return {
            "text": f"{product_data.title} {product_data.description}",
            "title": product_data.title,
            "description": product_data.description,
            "category": product_data.category,
            "brand": product_data.brand,
            "price": product_data.price,
            "images": product_data.images,
            "attributes": product_data.attributes,
            "reviews": product_data.reviews
        }
    
    def _create_task_description(self, task_type: str) -> str:
        """Create task-specific description."""
        task_descriptions = {
            "recommendation": "Extract features for product recommendation system focusing on user preferences and product similarity",
            "search": "Extract features for product search ranking focusing on relevance and popularity",
            "classification": "Extract features for product categorization and classification",
            "pricing": "Extract features for price prediction and optimization",
            "inventory": "Extract features for inventory management and demand forecasting"
        }
        
        return task_descriptions.get(task_type, "Extract general product features")
    
    def _extract_ecommerce_specific_features(self, product_data: ProductData) -> Dict[str, Any]:
        """Extract e-commerce specific features."""
        features = {}
        
        # Price-based features
        features["price_features"] = self._extract_price_features(product_data.price)
        
        # Category features
        features["category_features"] = self._extract_category_features(product_data.category)
        
        # Brand features
        features["brand_features"] = self._extract_brand_features(product_data.brand)
        
        # Review features
        features["review_features"] = self._extract_review_features(product_data.reviews)
        
        # Attribute features
        features["attribute_features"] = self._extract_attribute_features(product_data.attributes)
        
        # Inventory features
        features["inventory_features"] = self._extract_inventory_features(product_data.inventory)
        
        return features
    
    def _extract_price_features(self, price: float) -> Dict[str, float]:
        """Extract price-based features."""
        return {
            "price": price,
            "price_log": np.log(price + 1),
            "price_sqrt": np.sqrt(price),
            "price_category": self._categorize_price(price)
        }
    
    def _categorize_price(self, price: float) -> int:
        """Categorize price into buckets."""
        if price < 10:
            return 0  # Low
        elif price < 50:
            return 1  # Medium-low
        elif price < 200:
            return 2  # Medium
        elif price < 500:
            return 3  # Medium-high
        else:
            return 4  # High
    
    def _extract_category_features(self, category: str) -> Dict[str, Any]:
        """Extract category-based features."""
        # Simple category encoding
        category_hierarchy = category.split('/') if '/' in category else [category]
        
        return {
            "category": category,
            "category_depth": len(category_hierarchy),
            "main_category": category_hierarchy[0],
            "sub_category": category_hierarchy[-1] if len(category_hierarchy) > 1 else None,
            "category_encoded": hash(category) % 1000  # Simple hash encoding
        }
    
    def _extract_brand_features(self, brand: str) -> Dict[str, Any]:
        """Extract brand-based features."""
        return {
            "brand": brand,
            "brand_length": len(brand),
            "brand_encoded": hash(brand) % 1000,  # Simple hash encoding
            "is_premium": self._is_premium_brand(brand)
        }
    
    def _is_premium_brand(self, brand: str) -> bool:
        """Check if brand is premium (simplified)."""
        premium_brands = {
            "apple", "samsung", "sony", "nike", "adidas", "gucci", "louis vuitton",
            "chanel", "dior", "prada", "hermes", "rolex", "omega", "cartier"
        }
        return brand.lower() in premium_brands
    
    def _extract_review_features(self, reviews: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract features from product reviews."""
        if not reviews:
            return {
                "num_reviews": 0,
                "avg_rating": 0.0,
                "rating_std": 0.0,
                "sentiment_score": 0.0
            }
        
        ratings = [review.get("rating", 0) for review in reviews if "rating" in review]
        review_texts = [review.get("text", "") for review in reviews if "text" in review]
        
        features = {
            "num_reviews": len(reviews),
            "avg_rating": np.mean(ratings) if ratings else 0.0,
            "rating_std": np.std(ratings) if ratings else 0.0,
            "rating_min": min(ratings) if ratings else 0.0,
            "rating_max": max(ratings) if ratings else 0.0
        }
        
        # Simple sentiment analysis
        if review_texts:
            sentiment_scores = [self._analyze_sentiment(text) for text in review_texts]
            features["sentiment_score"] = np.mean(sentiment_scores)
            features["sentiment_std"] = np.std(sentiment_scores)
        else:
            features["sentiment_score"] = 0.0
            features["sentiment_std"] = 0.0
        
        return features
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis."""
        positive_words = {"good", "great", "excellent", "amazing", "love", "perfect"}
        negative_words = {"bad", "terrible", "awful", "hate", "worst", "disappointing"}
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _extract_attribute_features(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from product attributes."""
        features = {}
        
        for key, value in attributes.items():
            if isinstance(value, (int, float)):
                features[f"attr_{key}_numeric"] = value
            elif isinstance(value, str):
                features[f"attr_{key}_text"] = value
                features[f"attr_{key}_length"] = len(value)
            elif isinstance(value, bool):
                features[f"attr_{key}_bool"] = int(value)
        
        features["num_attributes"] = len(attributes)
        
        return features
    
    def _extract_inventory_features(self, inventory: int) -> Dict[str, Any]:
        """Extract inventory-based features."""
        return {
            "inventory": inventory,
            "inventory_log": np.log(inventory + 1),
            "is_in_stock": inventory > 0,
            "inventory_level": self._categorize_inventory(inventory)
        }
    
    def _categorize_inventory(self, inventory: int) -> int:
        """Categorize inventory level."""
        if inventory == 0:
            return 0  # Out of stock
        elif inventory < 10:
            return 1  # Low stock
        elif inventory < 100:
            return 2  # Medium stock
        else:
            return 3  # High stock


class ProductAnalyzer:
    """
    Product analysis utilities for e-commerce.
    
    Provides specialized analysis tools for product data
    including similarity, clustering, and recommendation scoring.
    """
    
    def __init__(self, pyrochain_instance):
        """Initialize with a PyroChain instance."""
        self.pyrochain = pyrochain_instance
        self.feature_extractor = EcommerceFeatureExtractor(pyrochain_instance)
    
    def analyze_product_similarity(
        self, 
        product1: ProductData, 
        product2: ProductData,
        feature_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Analyze similarity between two products."""
        # Extract features for both products
        features1 = self.feature_extractor.extract_product_features(product1)
        features2 = self.feature_extractor.extract_product_features(product2)
        
        # Calculate similarity scores
        similarity_scores = self._calculate_similarity_scores(features1, features2, feature_weights)
        
        return {
            "overall_similarity": similarity_scores["overall"],
            "feature_similarities": similarity_scores["features"],
            "recommendation_score": self._calculate_recommendation_score(similarity_scores),
            "product1_id": product1.product_id,
            "product2_id": product2.product_id
        }
    
    def _calculate_similarity_scores(
        self, 
        features1: Dict[str, Any], 
        features2: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Calculate similarity scores between feature sets."""
        if weights is None:
            weights = {
                "text": 0.3,
                "image": 0.2,
                "price": 0.2,
                "category": 0.15,
                "brand": 0.1,
                "reviews": 0.05
            }
        
        similarities = {}
        
        # Text similarity
        if "text_features" in features1 and "text_features" in features2:
            similarities["text"] = self._cosine_similarity(
                features1["text_features"], 
                features2["text_features"]
            )
        
        # Image similarity
        if "image_features" in features1 and "image_features" in features2:
            similarities["image"] = self._cosine_similarity(
                features1["image_features"], 
                features2["image_features"]
            )
        
        # Price similarity
        price1 = features1.get("ecommerce_features", {}).get("price_features", {}).get("price", 0)
        price2 = features2.get("ecommerce_features", {}).get("price_features", {}).get("price", 0)
        similarities["price"] = 1.0 - abs(price1 - price2) / max(price1, price2, 1)
        
        # Category similarity
        cat1 = features1.get("ecommerce_features", {}).get("category_features", {}).get("category", "")
        cat2 = features2.get("ecommerce_features", {}).get("category_features", {}).get("category", "")
        similarities["category"] = 1.0 if cat1 == cat2 else 0.0
        
        # Brand similarity
        brand1 = features1.get("ecommerce_features", {}).get("brand_features", {}).get("brand", "")
        brand2 = features2.get("ecommerce_features", {}).get("brand_features", {}).get("brand", "")
        similarities["brand"] = 1.0 if brand1 == brand2 else 0.0
        
        # Review similarity
        reviews1 = features1.get("ecommerce_features", {}).get("review_features", {})
        reviews2 = features2.get("ecommerce_features", {}).get("review_features", {})
        similarities["reviews"] = self._compare_review_features(reviews1, reviews2)
        
        # Calculate weighted overall similarity
        overall_similarity = sum(
            similarities.get(feature, 0) * weights.get(feature, 0)
            for feature in weights.keys()
        )
        
        return {
            "overall": overall_similarity,
            "features": similarities
        }
    
    def _cosine_similarity(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """Calculate cosine similarity between tensors."""
        if tensor1.shape != tensor2.shape:
            return 0.0
        
        # Flatten tensors
        vec1 = tensor1.flatten()
        vec2 = tensor2.flatten()
        
        # Calculate cosine similarity
        dot_product = torch.dot(vec1, vec2)
        norm1 = torch.norm(vec1)
        norm2 = torch.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return (dot_product / (norm1 * norm2)).item()
    
    def _compare_review_features(self, reviews1: Dict[str, float], reviews2: Dict[str, float]) -> float:
        """Compare review features between products."""
        if not reviews1 or not reviews2:
            return 0.0
        
        # Compare average ratings
        rating1 = reviews1.get("avg_rating", 0)
        rating2 = reviews2.get("avg_rating", 0)
        rating_similarity = 1.0 - abs(rating1 - rating2) / 5.0  # Assuming 5-star rating
        
        # Compare sentiment scores
        sentiment1 = reviews1.get("sentiment_score", 0)
        sentiment2 = reviews2.get("sentiment_score", 0)
        sentiment_similarity = 1.0 - abs(sentiment1 - sentiment2) / 2.0  # Assuming -1 to 1 range
        
        return (rating_similarity + sentiment_similarity) / 2.0
    
    def _calculate_recommendation_score(self, similarity_scores: Dict[str, Any]) -> float:
        """Calculate recommendation score based on similarity."""
        overall_similarity = similarity_scores["overall"]
        
        # Boost score for high similarity in key features
        if similarity_scores["features"].get("category", 0) > 0.8:
            overall_similarity *= 1.2
        
        if similarity_scores["features"].get("brand", 0) > 0.8:
            overall_similarity *= 1.1
        
        return min(1.0, overall_similarity)
    
    def find_similar_products(
        self, 
        target_product: ProductData, 
        product_catalog: List[ProductData],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar products in a catalog."""
        similarities = []
        
        for product in product_catalog:
            if product.product_id != target_product.product_id:
                similarity = self.analyze_product_similarity(target_product, product)
                similarities.append({
                    "product": product,
                    "similarity": similarity
                })
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x["similarity"]["overall_similarity"], reverse=True)
        
        return similarities[:top_k]
    
    def cluster_products(
        self, 
        products: List[ProductData], 
        n_clusters: int = 5
    ) -> Dict[str, Any]:
        """Cluster products based on their features."""
        from sklearn.cluster import KMeans
        
        # Extract features for all products
        features_list = []
        product_ids = []
        
        for product in products:
            features = self.feature_extractor.extract_product_features(product)
            if "raw_features" in features and isinstance(features["raw_features"], torch.Tensor):
                features_list.append(features["raw_features"].flatten().cpu().numpy())
                product_ids.append(product.product_id)
        
        if not features_list:
            return {"clusters": [], "error": "No valid features found"}
        
        # Perform clustering
        features_array = np.array(features_list)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_array)
        
        # Organize results
        clusters = {}
        for i, (product_id, label) in enumerate(zip(product_ids, cluster_labels)):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(product_id)
        
        return {
            "clusters": clusters,
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "inertia": kmeans.inertia_,
            "n_clusters": n_clusters
        }
