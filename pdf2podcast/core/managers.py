"""
Manager classes for LLM and TTS provider selection.
"""

from typing import Dict, Optional
from .llm import GeminiLLM
from .tts import AWSPollyTTS, GoogleTTS
from .base import BaseLLM, BaseTTS


class LLMManager:
    """
    Manager class for selecting and configuring LLM providers.
    """

    def __init__(self, provider_type: str, **kwargs):
        """
        Initialize LLM Manager.

        Args:
            provider_type (str): Type of LLM provider ("gemini", "openai", etc.)
            **kwargs: Configuration parameters for the selected provider
        """
        self.provider_type = provider_type
        self.config = kwargs

    def get_llm(self) -> BaseLLM:
        """
        Initialize and return the selected LLM provider.

        Returns:
            BaseLLM: Configured LLM instance

        Raises:
            ValueError: If provider_type is not supported
        """
        if self.provider_type == "gemini":
            return GeminiLLM(**self.config)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider_type}")


class TTSManager:
    """
    Manager class for selecting and configuring TTS providers.
    """

    def __init__(self, provider_type: str, **kwargs):
        """
        Initialize TTS Manager.

        Args:
            provider_type (str): Type of TTS provider ("aws", "google", etc.)
            **kwargs: Configuration parameters for the selected provider
        """
        self.provider_type = provider_type
        self.config = kwargs

    def get_tts(self) -> BaseTTS:
        """
        Initialize and return the selected TTS provider.

        Returns:
            BaseTTS: Configured TTS instance

        Raises:
            ValueError: If provider_type is not supported
        """
        if self.provider_type == "aws":
            return AWSPollyTTS(**self.config)
        elif self.provider_type == "google":
            return GoogleTTS(**self.config)
        else:
            raise ValueError(f"Unknown TTS provider: {self.provider_type}")
