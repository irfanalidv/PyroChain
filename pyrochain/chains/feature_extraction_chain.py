"""
Feature extraction chain for PyroChain.

Orchestrates the feature extraction process using LangChain chains
with memory capabilities for maintaining context across operations.
"""

from typing import Any

import torch
from langchain_core.messages import AIMessage, HumanMessage

from .base import BaseChain


class FeatureExtractionChain(BaseChain):
    """
    Chain for orchestrating feature extraction workflows.

    Manages the feature extraction process with memory-enabled
    context preservation and iterative refinement capabilities.
    """

    def __init__(
        self,
        agent,
        memory_type: str = "conversation_buffer",
        enable_iterative_refinement: bool = True,
        max_iterations: int = 3,
        **kwargs,
    ):
        """Initialize the feature extraction chain."""
        self.enable_iterative_refinement = enable_iterative_refinement
        self.max_iterations = max_iterations
        super().__init__(agent, memory_type, **kwargs)

    def run(
        self,
        input_data: dict[str, Any],
        task_description: str,
        feedback: str | None = None,
        iteration: int = 0,
    ) -> dict[str, Any]:
        """
        Run the feature extraction chain.

        Args:
            input_data: Input data to process
            task_description: Description of the feature extraction task
            feedback: Optional feedback from validation
            iteration: Current iteration number

        Returns:
            Dictionary containing extracted features and metadata
        """
        # Add context to memory
        self._add_context_to_memory(input_data, task_description, feedback, iteration)

        # Run the agent
        features = self.agent.run(
            input_data=input_data, task_description=task_description, feedback=feedback
        )

        # Add result to memory
        self._add_result_to_memory(features, iteration)

        # Check if iterative refinement is needed
        if self.enable_iterative_refinement and iteration < self.max_iterations:
            refinement_needed = self._check_refinement_needed(features)
            if refinement_needed:
                # Run another iteration
                refined_features = self.run(
                    input_data=input_data,
                    task_description=task_description,
                    feedback=self._generate_refinement_feedback(features),
                    iteration=iteration + 1,
                )
                return refined_features

        # Add final metadata
        features["chain_metadata"] = {
            "chain_type": "feature_extraction",
            "iteration": iteration,
            "iterative_refinement": self.enable_iterative_refinement,
            "memory_type": self.memory_type,
        }

        return features

    def _add_context_to_memory(
        self,
        input_data: dict[str, Any],
        task_description: str,
        feedback: str | None,
        iteration: int,
    ):
        """Add context to memory."""
        context_message = f"""
        Iteration {iteration} - Feature Extraction Task:
        Task: {task_description}
        Input Data: {self._summarize_input_data(input_data)}
        """

        if feedback:
            context_message += f"\nFeedback: {feedback}"

        self.add_to_memory(HumanMessage(content=context_message))

    def _add_result_to_memory(self, features: dict[str, Any], iteration: int):
        """Add result to memory."""
        raw = features.get("raw_features")
        num_vectors = len(raw) if raw is not None else 0  # raw_features may be None
        result_message = f"""
        Iteration {iteration} - Feature Extraction Result:
        Features extracted: {num_vectors} feature vectors
        Modalities: {features.get("metadata", {}).get("modalities", [])}
        Agent reasoning: {str(features.get("agent_reasoning") or "N/A")[:200]}...
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

    def _check_refinement_needed(self, features: dict[str, Any]) -> bool:
        """Check if feature refinement is needed."""
        # Simple heuristic for refinement
        if features.get("raw_features") is None:
            return True

        if (
            isinstance(features["raw_features"], torch.Tensor)
            and features["raw_features"].numel() == 0
        ):
            return True

        # Check if metadata is missing
        return bool(not features.get("metadata"))

    def _generate_refinement_feedback(self, features: dict[str, Any]) -> str:
        """Generate feedback for refinement."""
        feedback_parts = []

        if features.get("raw_features") is None:
            feedback_parts.append("No raw features extracted")

        if not features.get("metadata"):
            feedback_parts.append("Missing metadata")

        if not features.get("metadata", {}).get("modalities"):
            feedback_parts.append("No modalities detected")

        if feedback_parts:
            return "Please improve: " + "; ".join(feedback_parts)

        return "Features look good, but please double-check for completeness"

    def batch_extract_features(
        self, data_list: list[dict[str, Any]], task_description: str
    ) -> list[dict[str, Any]]:
        """Extract features for a batch of data samples."""
        results = []

        for i, data in enumerate(data_list):
            # Add batch context to memory
            batch_message = f"Processing batch item {i + 1}/{len(data_list)}"
            self.add_to_memory(HumanMessage(content=batch_message))

            # Extract features
            features = self.run(data, task_description)
            results.append(features)

        return results

    def extract_features_with_context(
        self,
        input_data: dict[str, Any],
        task_description: str,
        context_data: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Extract features with additional context data."""
        if context_data:
            # Add context data to memory
            context_message = f"Context data: {len(context_data)} samples provided for reference"
            self.add_to_memory(HumanMessage(content=context_message))

        return self.run(input_data, task_description)

    def get_extraction_history(self) -> list[dict[str, Any]]:
        """Get history of feature extractions."""
        messages = self.get_memory_messages()
        history = []

        current_extraction = {}
        for message in messages:
            content = str(message.content)
            if isinstance(message, HumanMessage) and "Feature Extraction Task" in content:
                current_extraction = {"input": content}
            elif isinstance(message, AIMessage) and "Feature Extraction Result" in content:
                current_extraction["output"] = content
                history.append(current_extraction)
                current_extraction = {}

        return history

    def refine_features(
        self, features: dict[str, Any], refinement_instructions: str
    ) -> dict[str, Any]:
        """Refine existing features based on instructions."""
        # Add refinement context to memory
        refinement_message = f"Refining features based on: {refinement_instructions}"
        self.add_to_memory(HumanMessage(content=refinement_message))

        # Create mock input data for refinement
        mock_input = {"refinement_target": features}

        # Run refinement
        refined_features = self.run(
            input_data=mock_input, task_description=f"Refine features: {refinement_instructions}"
        )

        return refined_features

    def get_chain_metrics(self) -> dict[str, Any]:
        """Get chain performance metrics."""
        history = self.get_extraction_history()

        if not history:
            return {"total_extractions": 0}

        total_extractions = len(history)
        successful_extractions = sum(
            1 for extraction in history if "Features extracted" in extraction.get("output", "")
        )

        return {
            "total_extractions": total_extractions,
            "successful_extractions": successful_extractions,
            "success_rate": (
                successful_extractions / total_extractions if total_extractions > 0 else 0
            ),
            "iterative_refinement_enabled": self.enable_iterative_refinement,
            "max_iterations": self.max_iterations,
        }
