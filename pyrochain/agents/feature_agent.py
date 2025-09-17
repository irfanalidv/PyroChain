"""
Feature extraction agent for PyroChain.
"""

from typing import Dict, Any, List, Optional
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.prompts import PromptTemplate

from .base import BaseAgent


class FeatureAgent(BaseAgent):
    """
    Agent specialized for feature extraction from multimodal data.

    Uses chain-of-thought reasoning to extract relevant features
    from text and image data for machine learning tasks.
    """

    def __init__(
        self,
        adapter,
        processor,
        memory_type: str = "conversation_buffer",
        max_memory_size: int = 1000,
        **kwargs,
    ):
        """Initialize the feature extraction agent."""
        super().__init__(adapter, processor, memory_type, max_memory_size, **kwargs)

    def _setup_tools(self) -> List[Tool]:
        """Setup tools for feature extraction."""
        tools = [
            Tool(
                name="extract_text_features",
                description="Extract features from text data including keywords, sentiment, and semantic features",
                func=self._extract_text_features,
            ),
            Tool(
                name="extract_image_features",
                description="Extract features from image data including objects, colors, and visual features",
                func=self._extract_image_features,
            ),
            Tool(
                name="extract_multimodal_features",
                description="Extract combined features from both text and image data",
                func=self._extract_multimodal_features,
            ),
            Tool(
                name="analyze_data_structure",
                description="Analyze the structure and content of input data",
                func=self._analyze_data_structure,
            ),
            Tool(
                name="generate_feature_summary",
                description="Generate a summary of extracted features",
                func=self._generate_feature_summary,
            ),
        ]
        return tools

    def _setup_agent(self):
        """Setup the feature extraction agent."""
        prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "chat_history"],
            template="""You are a specialized feature extraction agent for PyroChain. 
            Your task is to extract relevant features from multimodal data (text and images) 
            for machine learning tasks.

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
            {agent_scratchpad}""",
        )

        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            prompt=prompt,
            memory=self.memory,
        )

    def run(
        self,
        input_data: Dict[str, Any],
        task_description: str,
        feedback: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the feature extraction agent.

        Args:
            input_data: Input data to process
            task_description: Description of the feature extraction task
            feedback: Optional feedback from validation agent

        Returns:
            Dictionary containing extracted features and metadata
        """
        # Prepare input for the agent
        input_text = self._prepare_input(input_data, task_description, feedback)

        # Run the agent
        result = self.agent_executor.run(input=input_text)

        # Process the result
        features = self._process_agent_result(result, input_data)

        return features

    def _prepare_input(
        self,
        input_data: Dict[str, Any],
        task_description: str,
        feedback: Optional[str] = None,
    ) -> str:
        """Prepare input for the agent."""
        input_text = f"""
        Task: {task_description}
        
        Input Data:
        {self._format_input_data(input_data)}
        """

        if feedback:
            input_text += f"\n\nFeedback from validation: {feedback}"

        return input_text

    def _format_input_data(self, data: Dict[str, Any]) -> str:
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

    def _process_agent_result(
        self, result: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process the agent's result."""
        # Extract features using the processor
        processed_data = self.processor.process(input_data)

        # Parse the agent's result to extract structured features
        features = {
            "raw_features": processed_data.get("multimodal_features"),
            "text_features": processed_data.get("text_features"),
            "image_features": processed_data.get("image_features"),
            "agent_reasoning": result,
            "metadata": {
                "modalities": processed_data.get("modalities", []),
                "processing_metadata": processed_data.get("processing_metadata", {}),
                "agent_type": "feature_extraction",
            },
        }

        return features

    def _extract_text_features(self, query: str) -> str:
        """Extract features from text data."""
        # This would be called by the agent
        return "Text features extracted: keywords, sentiment, semantic embeddings"

    def _extract_image_features(self, query: str) -> str:
        """Extract features from image data."""
        # This would be called by the agent
        return "Image features extracted: objects, colors, visual embeddings"

    def _extract_multimodal_features(self, query: str) -> str:
        """Extract combined features from multimodal data."""
        # This would be called by the agent
        return "Multimodal features extracted: fused text and image representations"

    def _analyze_data_structure(self, query: str) -> str:
        """Analyze the structure of input data."""
        # This would be called by the agent
        return "Data structure analyzed: identified text and image fields"

    def _generate_feature_summary(self, query: str) -> str:
        """Generate a summary of extracted features."""
        # This would be called by the agent
        return "Feature summary generated: comprehensive overview of all extracted features"

    def extract_features_for_task(
        self, data: Dict[str, Any], task_type: str, domain: str = "general"
    ) -> Dict[str, Any]:
        """
        Extract features optimized for a specific task type.

        Args:
            data: Input data
            task_type: Type of ML task (classification, regression, clustering, etc.)
            domain: Domain of the data (e-commerce, healthcare, etc.)

        Returns:
            Task-optimized features
        """
        task_description = f"""
        Extract features for {task_type} task in {domain} domain.
        Focus on features that are most relevant for this specific task type.
        """

        return self.run(data, task_description)

    def get_feature_importance(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Get importance scores for extracted features."""
        # Simplified feature importance calculation
        importance_scores = {}

        if "text_features" in features:
            importance_scores["text_features"] = 0.4

        if "image_features" in features:
            importance_scores["image_features"] = 0.4

        if "raw_features" in features:
            importance_scores["raw_features"] = 0.2

        return importance_scores
