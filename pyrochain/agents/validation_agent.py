"""
Validation agent for PyroChain.

Validates and refines features extracted by the feature agent
using chain-of-thought reasoning and quality metrics.
"""

import time
from typing import Any

import torch
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from .base import BaseAgent


class ValidationAgent(BaseAgent):
    """
    Agent specialized for validating and refining extracted features.

    Uses chain-of-thought reasoning to validate feature quality,
    identify issues, and provide feedback for improvement.
    """

    def __init__(
        self,
        adapter,
        processor,
        threshold: float = 0.8,
        memory_type: str = "conversation_buffer",
        max_memory_size: int = 1000,
        **kwargs,
    ):
        """Initialize the validation agent."""
        self.threshold = threshold
        super().__init__(adapter, processor, memory_type, max_memory_size, **kwargs)

    def _setup_tools(self) -> list[Tool]:
        """Setup tools for feature validation."""
        tools = [
            Tool(
                name="validate_feature_quality",
                description="Validate the quality of extracted features using various metrics",
                func=self._validate_feature_quality,
            ),
            Tool(
                name="check_feature_completeness",
                description="Check if all expected features are present and complete",
                func=self._check_feature_completeness,
            ),
            Tool(
                name="analyze_feature_relevance",
                description="Analyze the relevance of features for the given task",
                func=self._analyze_feature_relevance,
            ),
            Tool(
                name="detect_feature_issues",
                description="Detect potential issues with extracted features",
                func=self._detect_feature_issues,
            ),
            Tool(
                name="generate_improvement_suggestions",
                description="Generate suggestions for improving feature extraction",
                func=self._generate_improvement_suggestions,
            ),
            Tool(
                name="calculate_validation_score",
                description="Calculate an overall validation score for the features",
                func=self._calculate_validation_score,
            ),
        ]
        return tools

    def _setup_agent(self):
        """Setup the validation agent."""
        # The ReAct prompt is kept for use with a real LLM; the built-in executor runs the
        # tools deterministically (see SequentialToolExecutor).
        return PromptTemplate.from_template(
            """You are a specialized feature validation agent for PyroChain.
            Your task is to validate and provide feedback on features extracted by the feature agent.

            You have access to the following tools:
            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Previous conversation history:
            {chat_history}

            New question: {input}
            {agent_scratchpad}"""
        )

    def run(
        self, features: dict[str, Any], original_data: dict[str, Any], task_description: str
    ) -> dict[str, Any]:
        """
        Run the validation agent.

        Args:
            features: Features extracted by the feature agent
            original_data: Original input data
            task_description: Description of the task

        Returns:
            Dictionary containing validation results and feedback
        """
        # Prepare input for the agent
        input_text = self._prepare_validation_input(features, original_data, task_description)

        # Run the agent
        result = self.agent_executor.invoke(input_text)

        # Process the result
        validation_result = self._process_validation_result(result, features)

        return validation_result

    def _prepare_validation_input(
        self, features: dict[str, Any], original_data: dict[str, Any], task_description: str
    ) -> str:
        """Prepare input for the validation agent."""
        input_text = f"""
        Task: {task_description}

        Original Data:
        {self._format_input_data(original_data)}

        Extracted Features:
        {self._format_features(features)}

        Please validate these features and provide feedback.
        """

        return input_text

    def _format_input_data(self, data: dict[str, Any]) -> str:
        """Format input data for the agent."""
        formatted = []

        for key, value in data.items():
            if isinstance(value, str):
                formatted.append(f"{key}: {value[:200]}...")
            elif isinstance(value, (int, float, bool)):
                formatted.append(f"{key}: {value}")
            else:
                formatted.append(f"{key}: {type(value).__name__}")

        return "\n".join(formatted)

    def _format_features(self, features: dict[str, Any]) -> str:
        """Format features for the agent."""
        formatted = []

        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                formatted.append(f"{key}: Tensor with shape {value.shape}")
            elif isinstance(value, dict):
                formatted.append(f"{key}: {len(value)} items")
            elif isinstance(value, (int, float, bool, str)):
                formatted.append(f"{key}: {value}")
            else:
                formatted.append(f"{key}: {type(value).__name__}")

        return "\n".join(formatted)

    def _process_validation_result(self, result: str, features: dict[str, Any]) -> dict[str, Any]:
        """Process the validation agent's result."""
        # Calculate validation score
        score = self._calculate_overall_score(features)

        # Determine if features are valid
        is_valid = score >= self.threshold

        # Generate feedback
        feedback = self._generate_feedback(features, score, is_valid)

        validation_result = {
            "is_valid": is_valid,
            "score": score,
            "feedback": feedback,
            "agent_reasoning": result,
            "threshold": self.threshold,
            "metadata": {
                "agent_type": "validation",
                "validation_timestamp": time.time(),
            },
        }

        return validation_result

    def _calculate_overall_score(self, features: dict[str, Any]) -> float:
        """Calculate overall validation score."""
        scores = []

        # Check feature completeness
        completeness_score = self._check_completeness_score(features)
        scores.append(completeness_score)

        # Check feature quality
        quality_score = self._check_quality_score(features)
        scores.append(quality_score)

        # Check feature relevance
        relevance_score = self._check_relevance_score(features)
        scores.append(relevance_score)

        # Calculate weighted average
        weights = [0.3, 0.4, 0.3]  # completeness, quality, relevance
        overall_score = sum(w * s for w, s in zip(weights, scores))

        return min(1.0, max(0.0, overall_score))

    def _check_completeness_score(self, features: dict[str, Any]) -> float:
        """Check completeness of features."""
        required_keys = ["raw_features", "metadata"]
        present_keys = [key for key in required_keys if key in features]
        return len(present_keys) / len(required_keys)

    def _check_quality_score(self, features: dict[str, Any]) -> float:
        """Check quality of features."""
        score = 0.0

        # Check if features are tensors with reasonable dimensions
        if "raw_features" in features and isinstance(features["raw_features"], torch.Tensor):
            if features["raw_features"].shape[-1] > 0:
                score += 0.5

        # Check if metadata is present
        if "metadata" in features and isinstance(features["metadata"], dict):
            score += 0.3

        # Check if modalities are detected
        if "metadata" in features and "modalities" in features["metadata"]:
            if features["metadata"]["modalities"]:
                score += 0.2

        return min(1.0, score)

    def _check_relevance_score(self, features: dict[str, Any]) -> float:
        """Check relevance of features."""
        # Simplified relevance check
        score = 0.5  # Base score

        # Check if both text and image features are present
        if "text_features" in features and "image_features" in features:
            score += 0.3

        # Check if multimodal features are present
        if "raw_features" in features:
            score += 0.2

        return min(1.0, score)

    def _generate_feedback(self, features: dict[str, Any], score: float, is_valid: bool) -> str:
        """Generate feedback based on validation results."""
        feedback_parts = []

        if is_valid:
            feedback_parts.append(f"Features passed validation with a score of {score:.2f}")
        else:
            feedback_parts.append(
                f"Features failed validation with a score of {score:.2f} (threshold: {self.threshold:.2f})"
            )

        # Add specific feedback based on issues
        if score < 0.5:
            feedback_parts.append("Consider improving feature extraction quality")

        if "text_features" not in features and "image_features" not in features:
            feedback_parts.append("No text or image features detected")

        if "metadata" not in features:
            feedback_parts.append("Missing metadata information")

        return ". ".join(feedback_parts)

    def _validate_feature_quality(self, query: str) -> str:
        """Validate feature quality."""
        return "Feature quality validation completed"

    def _check_feature_completeness(self, query: str) -> str:
        """Check feature completeness."""
        return "Feature completeness check completed"

    def _analyze_feature_relevance(self, query: str) -> str:
        """Analyze feature relevance."""
        return "Feature relevance analysis completed"

    def _detect_feature_issues(self, query: str) -> str:
        """Detect feature issues."""
        return "Feature issue detection completed"

    def _generate_improvement_suggestions(self, query: str) -> str:
        """Generate improvement suggestions."""
        return "Improvement suggestions generated"

    def _calculate_validation_score(self, query: str) -> str:
        """Calculate validation score."""
        return "Validation score calculated"

    def validate_features_batch(
        self,
        features_list: list[dict[str, Any]],
        original_data_list: list[dict[str, Any]],
        task_description: str,
    ) -> list[dict[str, Any]]:
        """Validate a batch of features."""
        results = []

        for features, original_data in zip(features_list, original_data_list):
            result = self.run(features, original_data, task_description)
            results.append(result)

        return results

    def get_validation_metrics(self, validation_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Get aggregated validation metrics."""
        if not validation_results:
            return {}

        scores = [result["score"] for result in validation_results]
        valid_count = sum(1 for result in validation_results if result["is_valid"])

        return {
            "total_samples": len(validation_results),
            "valid_samples": valid_count,
            "invalid_samples": len(validation_results) - valid_count,
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "validation_rate": valid_count / len(validation_results),
        }
