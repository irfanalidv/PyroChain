"""
Validation chain for PyroChain.

Orchestrates the feature validation process using LangChain chains
with memory capabilities for maintaining validation context and history.
"""

from typing import Any

import torch
from langchain_core.messages import AIMessage, HumanMessage

from .base import BaseChain


class ValidationChain(BaseChain):
    """
    Chain for orchestrating feature validation workflows.

    Manages the validation process with memory-enabled context
    preservation and feedback generation capabilities.
    """

    def __init__(
        self,
        agent,
        memory_type: str = "conversation_buffer",
        enable_feedback_loop: bool = True,
        **kwargs,
    ):
        """Initialize the validation chain."""
        self.enable_feedback_loop = enable_feedback_loop
        super().__init__(agent, memory_type, **kwargs)

    def run(
        self, features: dict[str, Any], original_data: dict[str, Any], task_description: str
    ) -> dict[str, Any]:
        """
        Run the validation chain.

        Args:
            features: Features to validate
            original_data: Original input data
            task_description: Description of the task

        Returns:
            Dictionary containing validation results and feedback
        """
        # Add validation context to memory
        self._add_validation_context_to_memory(features, original_data, task_description)

        # Run the validation agent
        validation_result = self.agent.run(
            features=features, original_data=original_data, task_description=task_description
        )

        # Add result to memory
        self._add_validation_result_to_memory(validation_result)

        # Generate additional feedback if needed
        if self.enable_feedback_loop and not validation_result.get("is_valid", False):
            additional_feedback = self._generate_additional_feedback(validation_result, features)
            validation_result["additional_feedback"] = additional_feedback

        # Add chain metadata
        validation_result["chain_metadata"] = {
            "chain_type": "validation",
            "feedback_loop_enabled": self.enable_feedback_loop,
            "memory_type": self.memory_type,
        }

        return validation_result

    def _add_validation_context_to_memory(
        self, features: dict[str, Any], original_data: dict[str, Any], task_description: str
    ):
        """Add validation context to memory."""
        context_message = f"""
        Validation Task:
        Task: {task_description}
        Original Data: {self._summarize_input_data(original_data)}
        Features to Validate: {self._summarize_features(features)}
        """

        self.add_to_memory(HumanMessage(content=context_message))

    def _add_validation_result_to_memory(self, validation_result: dict[str, Any]):
        """Add validation result to memory."""
        result_message = f"""
        Validation Result:
        Valid: {validation_result.get("is_valid", False)}
        Score: {validation_result.get("score", 0.0):.2f}
        Feedback: {validation_result.get("feedback", "N/A")[:200]}...
        """

        self.add_to_memory(AIMessage(content=result_message))

    def _summarize_input_data(self, input_data: dict[str, Any]) -> str:
        """Summarize input data for memory."""
        summary_parts = []

        for key, value in input_data.items():
            if isinstance(value, str):
                summary_parts.append(f"{key}: {value[:100]}...")
            elif isinstance(value, (int, float, bool)):
                summary_parts.append(f"{key}: {value}")
            else:
                summary_parts.append(f"{key}: {type(value).__name__}")

        return "; ".join(summary_parts)

    def _summarize_features(self, features: dict[str, Any]) -> str:
        """Summarize features for memory."""
        summary_parts = []

        if "raw_features" in features:
            if isinstance(features["raw_features"], torch.Tensor):
                summary_parts.append(f"raw_features: Tensor{features['raw_features'].shape}")
            else:
                summary_parts.append(f"raw_features: {type(features['raw_features']).__name__}")

        if "metadata" in features:
            modalities = features["metadata"].get("modalities", [])
            summary_parts.append(f"modalities: {modalities}")

        if "text_features" in features:
            summary_parts.append("text_features: present")

        if "image_features" in features:
            summary_parts.append("image_features: present")

        return "; ".join(summary_parts)

    def _generate_additional_feedback(
        self, validation_result: dict[str, Any], features: dict[str, Any] | None = None
    ) -> str:
        """Generate additional feedback for failed validations.

        ``features`` are the features that were validated. (The validation result has no
        ``"features"`` key, so the checks below used to fire on every failed validation.)
        """
        features = features if features is not None else validation_result.get("features", {})
        feedback_parts = []

        score = validation_result.get("score", 0.0)

        if score < 0.3:
            feedback_parts.append("Consider completely re-extracting features")
        elif score < 0.6:
            feedback_parts.append("Significant improvements needed in feature extraction")
        else:
            feedback_parts.append("Minor improvements needed")

        # Add specific suggestions based on common issues
        if features.get("raw_features") is None:
            feedback_parts.append("Ensure raw features are extracted")

        if not (features.get("metadata") or {}).get("modalities"):
            feedback_parts.append("Ensure modalities are properly detected")

        return ". ".join(feedback_parts)

    def batch_validate_features(
        self,
        features_list: list[dict[str, Any]],
        original_data_list: list[dict[str, Any]],
        task_description: str,
    ) -> list[dict[str, Any]]:
        """Validate a batch of features."""
        results = []

        for i, (features, original_data) in enumerate(
            zip(features_list, original_data_list, strict=True)
        ):
            # Add batch context to memory
            batch_message = f"Validating batch item {i + 1}/{len(features_list)}"
            self.add_to_memory(HumanMessage(content=batch_message))

            # Validate features
            validation_result = self.run(features, original_data, task_description)
            results.append(validation_result)

        return results

    def validate_with_historical_context(
        self,
        features: dict[str, Any],
        original_data: dict[str, Any],
        task_description: str,
        historical_validations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Validate features with historical context."""
        if historical_validations:
            # Add historical context to memory
            context_message = (
                f"Historical context: {len(historical_validations)} previous validations available"
            )
            self.add_to_memory(HumanMessage(content=context_message))

        return self.run(features, original_data, task_description)

    def get_validation_history(self) -> list[dict[str, Any]]:
        """Get history of validations."""
        messages = self.get_memory_messages()
        history = []

        current_validation = {}
        for message in messages:
            content = str(message.content)
            if isinstance(message, HumanMessage) and "Validation Task" in content:
                current_validation = {"input": content}
            elif isinstance(message, AIMessage) and "Validation Result" in content:
                current_validation["output"] = content
                history.append(current_validation)
                current_validation = {}

        return history

    def get_validation_metrics(self) -> dict[str, Any]:
        """Get validation performance metrics."""
        history = self.get_validation_history()

        if not history:
            return {"total_validations": 0}

        total_validations = len(history)

        # Parse validation results from history
        valid_count = 0
        scores = []

        for validation in history:
            output = validation.get("output", "")
            if "Valid: True" in output:
                valid_count += 1

            # Extract score (simplified parsing)
            if "Score:" in output:
                try:
                    score_line = next(line for line in output.split("\n") if "Score:" in line)
                    score = float(score_line.split(":")[1].strip())
                    scores.append(score)
                except (ValueError, IndexError):
                    pass

        return {
            "total_validations": total_validations,
            "valid_count": valid_count,
            "invalid_count": total_validations - valid_count,
            "validation_rate": valid_count / total_validations if total_validations > 0 else 0,
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
        }

    def generate_validation_report(self) -> dict[str, Any]:
        """Generate a comprehensive validation report."""
        metrics = self.get_validation_metrics()
        history = self.get_validation_history()

        report = {
            "summary": metrics,
            "recent_validations": history[-5:] if history else [],  # Last 5 validations
            "recommendations": self._generate_recommendations(metrics),
            "chain_state": self.get_chain_state(),
        }

        return report

    def _generate_recommendations(self, metrics: dict[str, Any]) -> list[str]:
        """Generate recommendations based on validation metrics."""
        recommendations = []

        validation_rate = metrics.get("validation_rate", 0.0)
        average_score = metrics.get("average_score", 0.0)

        if validation_rate < 0.5:
            recommendations.append("Consider improving feature extraction quality")

        if average_score < 0.6:
            recommendations.append("Focus on feature completeness and relevance")

        if metrics.get("total_validations", 0) < 10:
            recommendations.append("Run more validations to get better statistics")

        return recommendations
