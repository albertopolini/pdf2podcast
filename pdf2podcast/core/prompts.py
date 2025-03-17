"""
Prompt templates and mappings for podcast generation.
"""

from typing import Dict, Any

from pdf2podcast.core.base import BasePromptBuilder

# Detailed complexity mappings
COMPLEXITY_MAPPING = {
    "simple": {
        "vocabulary": "Use basic terms with clear explanations of any technical concepts",
        "structure": "Present ideas in a straightforward, linear progression",
        "examples": "Illustrate points with everyday scenarios and familiar contexts",
        "depth": "Focus on foundational concepts and practical understanding",
        "style": "Conversational and approachable, like explaining to a friend",
    },
    "intermediate": {
        "vocabulary": "Balance technical terms with clear explanations",
        "structure": "Build concepts progressively with logical connections",
        "examples": "Use industry-relevant applications and practical cases",
        "depth": "Explore interconnections between concepts",
        "style": "Professional but accessible, like a knowledgeable mentor",
    },
    "advanced": {
        "vocabulary": "Employ proper technical terminology",
        "structure": "Present complex relationships and detailed analysis",
        "examples": "Include specialized cases and technical applications",
        "depth": "Cover theoretical aspects and advanced implications",
        "style": "Technical and precise, like an expert presentation",
    },
}

# Target audience adaptations
AUDIENCE_MAPPING = {
    "general": {
        "background": "No specific technical background",
        "focus": "Practical understanding and real-world relevance",
        "terminology": "Explain all technical terms",
        "examples": "Common, everyday applications",
        "engagement": "Emphasis on general interest and practical value",
    },
    "students": {
        "background": "Academic context with basic field knowledge",
        "focus": "Systematic learning and fundamental principles",
        "terminology": "Build technical vocabulary with explanations",
        "examples": "Study-relevant scenarios and academic applications",
        "engagement": "Educational approach with clear learning objectives",
    },
    "professionals": {
        "background": "Working knowledge of the field",
        "focus": "Practical applications and industry relevance",
        "terminology": "Industry-standard terms and concepts",
        "examples": "Business cases and professional scenarios",
        "engagement": "Emphasis on practical implementation and value",
    },
    "experts": {
        "background": "Deep domain knowledge",
        "focus": "Advanced concepts and theoretical implications",
        "terminology": "Specialized technical vocabulary",
        "examples": "Complex case studies and cutting-edge applications",
        "engagement": "High-level technical discussion",
    },
    "enthusiasts": {
        "background": "Interest-driven basic knowledge",
        "focus": "Interesting aspects and hobby applications",
        "terminology": "Mix of basic and intermediate terms",
        "examples": "Hobby-relevant scenarios and DIY applications",
        "engagement": "Focus on interesting aspects and practical projects",
    },
}


class PodcastPromptTemplate:
    """Template provider for podcast generation prompts."""

    @staticmethod
    def get_base_prompt(
        text: str,
        complexity: str,
        target_audience: str,
        min_length: int,
        **kwargs: Dict[str, Any],
    ) -> str:
        """
        Get the base prompt for podcast script generation.

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
        Generate a podcast script using ONLY the provided text content. Follow these STRICT requirements:

        CRITICAL - DO NOT INCLUDE:
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

        Content Adaptation:
        1. Complexity Level ({complexity}):
           - Vocabulary: {complexity_settings['vocabulary']}
           - Structure: {complexity_settings['structure']}
           - Examples: {complexity_settings['examples']}
           - Depth: {complexity_settings['depth']}
           - Style: {complexity_settings['style']}

        2. Target Audience ({target_audience}):
           - Expected Background: {audience_settings['background']}
           - Content Focus: {audience_settings['focus']}
           - Terminology Usage: {audience_settings['terminology']}
           - Example Types: {audience_settings['examples']}
           - Engagement Style: {audience_settings['engagement']}

        Structure ({min_length} characters total):
        1. Direct Content Start (15%):
           - Begin with clear context for the audience
           - Frame the topic appropriately
           - Set proper expectations

        2. Main Explanation (65%):
           - Adapt depth to audience and complexity level
           - Build understanding progressively
           - Use appropriate terminology
           - Provide relevant context

        3. Real Applications (10%):
           - Examples matching audience interests
           - Practical scenarios at right complexity
           - Relevant use cases

        4. Conclusion (10%):
           - Summarize at appropriate level
           - Reinforce key points
           - Maintain technical accuracy

        Content Requirements:
        - Use ONLY information from source text
        - Maintain consistent complexity level
        - Match audience expectations
        - Clear verbal descriptions
        - Natural transitions
        - Pure narration style
        - Focus on substance
        - No external examples

        Source text:
        {text}

        Generate a focused, well-adapted script that strictly follows these requirements.
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
        Get the prompt for expanding an existing script.

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
        Expand this script to at least {min_length} characters.
        CRITICAL - Maintain all previous rules:
        - NO sound effects, music, or audio cues
        - NO intro/outro elements
        - NO references to podcast format
        - Focus ONLY on content from source text
        - Keep {complexity} level complexity
        - Target {target_audience} audience
        - Maintain pure narration style
        
        Add more detail and examples appropriate for the audience.

        Current script:
        {script}
        """


class PodcastPromptBuilder(BasePromptBuilder):
    """Prompt builder for podcast script generation."""

    def __init__(self, template_provider=None):
        """
        Initialize with optional custom template provider.

        Args:
            template_provider: Template provider class (default: PodcastPromptTemplate)
        """
        if template_provider is None:
            from .prompts import PodcastPromptTemplate

            template_provider = PodcastPromptTemplate
        self.templates = template_provider

    def build_prompt(self, text: str, **kwargs) -> str:
        """Build main generation prompt."""
        return self.templates.get_base_prompt(text, **kwargs)

    def build_expand_prompt(self, text: str, **kwargs) -> str:
        """Build expansion prompt."""
        return self.templates.get_expand_prompt(text, **kwargs)
