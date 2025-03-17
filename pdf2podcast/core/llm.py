"""
Large Language Model (LLM) implementations for pdf2podcast.
"""

import os
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from .base import BaseLLM
from .prompts import PodcastPromptBuilder


class GeminiLLM(BaseLLM):
    """
    Google's Gemini-based LLM implementation with optimized content generation.
    """

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_output_tokens: int = 4096,
        streaming: bool = False,
        prompt_builder: PodcastPromptBuilder = None,
    ):
        """
        Initialize Gemini LLM system.

        Args:
            api_key (str, optional): Google API key. If not provided, will look for GENAI_API_KEY env var
            model_name (str): Name of the Gemini model to use (default: "gemini-1.5-flash")
            temperature (float): Sampling temperature (default: 0.2)
            top_p (float): Nucleus sampling parameter (default: 0.9)
            max_output_tokens (int): Maximum output length (default: 4096)
            streaming (bool): Whether to use streaming mode (default: False)
            prompt_builder (Optional[PodcastPromptBuilder]): Custom prompt builder
        """
        super().__init__(prompt_builder or PodcastPromptBuilder())

        if api_key is None:
            load_dotenv()
            api_key = os.getenv("GENAI_API_KEY")
            if not api_key:
                raise ValueError("No API key provided and GENAI_API_KEY not found")

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            streaming=streaming,
            google_api_key=api_key,
        )

    def _clean_text(self, text: str) -> str:
        """Clean text using regex patterns to remove visual references."""
        patterns = [
            r"(Figure|Fig\.|Table|Image)\s+\d+[a-z]?",
            r"(shown|illustrated|depicted|as seen) (in|on|above|below)",
            r"(refer to|see|view) (figure|table|image)",
            r"\(fig\.\s*\d+\)",
            r"as (shown|depicted) (here|below|above)",
        ]

        processed = text
        for pattern in patterns:
            processed = re.sub(pattern, "", processed, flags=re.IGNORECASE)

        processed = re.sub(r"\s+", " ", processed)
        return processed.strip()

    def generate_podcast_script(
        self,
        text: str,
        complexity: str = "intermediate",
        target_audience: str = "general",
        min_length: int = 10000,
        **kwargs: Dict[str, Any],
    ) -> str:
        """
        Generate a coherent podcast script adapted to target audience and complexity.

        Args:
            text (str): Input text to convert into a podcast script
            complexity (str): Desired complexity level
            target_audience (str): Target audience category
            min_length (int): Minimum target length in characters
            **kwargs: Additional parameters

        Returns:
            str: Generated podcast script
        """
        try:
            # Clean text
            processed_text = self._clean_text(text)

            # Generate initial script
            prompt = self.prompt_builder.build_prompt(
                text=processed_text,
                complexity=complexity,
                target_audience=target_audience,
                min_length=min_length,
                **kwargs,
            )

            response = self.llm.invoke(prompt)
            script = response.content.strip()

            # Expand if needed
            if len(script) < min_length:
                expand_prompt = self.prompt_builder.build_expand_prompt(
                    text=script,
                    complexity=complexity,
                    target_audience=target_audience,
                    min_length=min_length,
                    **kwargs,
                )
                response = self.llm.invoke(expand_prompt)
                script = response.content.strip()

            return script

        except Exception as e:
            raise Exception(f"Failed to generate podcast script: {str(e)}")
