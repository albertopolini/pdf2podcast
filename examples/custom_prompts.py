"""
Custom prompt template and builder for storytelling-style podcast generation.
"""

from typing import Dict, Any
from pdf2podcast.core.base import BasePromptBuilder
from pdf2podcast.core.prompts import COMPLEXITY_MAPPING, AUDIENCE_MAPPING


class StorytellingPromptTemplate:
    """Template provider for storytelling-style prompts."""

    @staticmethod
    def get_base_prompt(
        text: str,
        complexity: str,
        target_audience: str,
        min_length: int,
        **kwargs: Dict[str, Any],
    ) -> str:
        """
        Get the base prompt for storytelling-style podcast generation.

        Args:
            text (str): Source text
            complexity (str): Desired complexity level
            target_audience (str): Target audience category
            min_length (int): Minimum target length
            **kwargs: Additional parameters

        Returns:
            str: Formatted prompt
        """
        complexity_settings = COMPLEXITY_MAPPING.get(
            complexity, COMPLEXITY_MAPPING["intermediate"]
        )
        audience_settings = AUDIENCE_MAPPING.get(
            target_audience, AUDIENCE_MAPPING["general"]
        )

        return f"""
        Transform this technical content into an engaging narrative. Follow these requirements:

        CRITICAL - Maintain all standard podcast rules:
        - NO sound effects (whoosh, ding, etc.)
        - NO music or jingles
        - NO audio transitions ("fade in", "fade out", etc.)
        - NO audio instructions or cues
        - NO intro/outro music references
        - NO host introductions or sign-offs
        - NO references to audio elements
        - NO sound descriptions in parentheses
        - NO "welcome" or "thanks for listening" phrases
        - NO podcast name or branding
        - NO references to figures, diagrams, or visual elements

        Storytelling Structure ({min_length} characters):
        1. Hook/Opening Scene (10%):
           - Start with a compelling scenario or question
           - Frame the technical content in a relatable context
           - Create curiosity about the subject

        2. Problem/Challenge Setup (20%):
           - Present the technical concepts as challenges to solve
           - Build tension around the problems in the field
           - Create a sense of discovery

        3. Journey Through Solutions (50%):
           - Explain technical concepts as revelations
           - Present research findings as plot developments
           - Build connections between ideas dramatically
           - Maintain {complexity_settings['depth']} while using narrative flow
           - Use language appropriate for {audience_settings['background']}

        4. Impact & Resolution (20%):
           - Show how the solutions affect the field
           - Connect to {audience_settings['focus']}
           - Create satisfying conclusions

        Style Guidelines:
        - Use narrative techniques: foreshadowing, callbacks, revelations
        - Create character-like elements from concepts
        - Build tension and resolution cycles
        - Maintain technical accuracy while being engaging
        - Focus on progression and discovery
        - Use {complexity_settings['vocabulary']}
        - Provide examples relevant to {audience_settings['examples']}

        Source text:
        {text}
        """

    @staticmethod
    def get_expand_prompt(
        script: str,
        min_length: int,
        complexity: str,
        target_audience: str,
        **kwargs: Dict[str, Any],
    ) -> str:
        """
        Get the prompt for expanding an existing storytelling script.

        Args:
            script (str): Current script
            min_length (int): Target minimum length
            complexity (str): Desired complexity level
            target_audience (str): Target audience category
            **kwargs: Additional parameters

        Returns:
            str: Formatted expansion prompt
        """
        return f"""
        Expand this narrative while maintaining the storytelling approach.
        Target length: {min_length} characters
        
        Guidelines:
        - Deepen the narrative elements
        - Add more detailed scenarios
        - Enhance the progression of ideas
        - Keep {complexity} complexity level
        - Stay focused on {target_audience} audience
        - NO audio elements or sound effects
        
        Current script:
        {script}
        """


class StorytellingPromptBuilder(BasePromptBuilder):
    """Builder for storytelling-style prompts."""

    def __init__(self):
        """Initialize the storytelling prompt builder."""
        self.templates = StorytellingPromptTemplate

    def build_prompt(self, text: str, **kwargs) -> str:
        """Build main generation prompt for storytelling style."""
        return self.templates.get_base_prompt(text, **kwargs)

    def build_expand_prompt(self, text: str, **kwargs) -> str:
        """Build expansion prompt for storytelling style."""
        return self.templates.get_expand_prompt(text, **kwargs)
