"""
Manager classes for LLM and TTS provider selection.
"""

from typing import Dict, Optional, Any
import logging
from pydantic import BaseModel, Field, field_validator

# Setup logging
logger = logging.getLogger(__name__)


class LLMConfig(BaseModel):
    """Configuration model for LLM providers."""

    api_key: Optional[str] = Field(None, description="API key for the provider")
    model_name: Optional[str] = None
    temperature: float = Field(0.2, ge=0.0, le=1.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_output_tokens: int = Field(4096, gt=0)
    streaming: bool = False


class TTSConfig(BaseModel):
    """Configuration model for TTS providers."""

    voice_id: Optional[str] = None
    language: Optional[str] = None
    region_name: Optional[str] = None
    engine: str = Field("neural", pattern="^(standard|neural)$")
    temp_dir: str = "temp"

    @field_validator("region_name")
    def validate_region(cls, v, values):
        if values.get("tts_provider") == "aws" and not v:
            raise ValueError("region_name is required for AWS Polly")
        return v


from .llm import GeminiLLM
from .tts import AWSPollyTTS, GoogleTTS
from .base import BaseLLM, BaseTTS


class LLMManager:
    """
    Manager class for selecting and configuring LLM providers.
    """

    def __init__(self, llm_provider: str, **kwargs: Dict[str, Any]):
        """
        Initialize LLM Manager.

        Args:
            llm_provider (str): Type of LLM provider ("gemini", "openai", etc.)
            **kwargs: Configuration parameters for the selected provider
        """
        # Validate configuration
        if llm_provider not in ["gemini", "openai"]:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        self.llm_provider = llm_provider

        try:
            config = LLMConfig(**kwargs)
            self.config = config.model_dump(exclude_unset=True)
            logger.info(f"Initialized LLM Manager with provider: {llm_provider}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM Manager: {str(e)}")
            raise ValueError(f"Invalid LLM configuration: {str(e)}")

    def get_llm(self) -> Optional[BaseLLM]:
        """
        Initialize and return the selected LLM provider.

        Returns:
            BaseLLM: Configured LLM instance

        Raises:
            ValueError: If llm_provider is not supported
        """
        try:
            if self.llm_provider == "gemini":
                return GeminiLLM(**self.config)
            # Add support for other providers here
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM instance: {str(e)}")
            raise


class TTSManager:
    """
    Manager class for selecting and configuring TTS providers.
    """

    def __init__(self, tts_provider: str, **kwargs: Dict[str, Any]):
        """
        Initialize TTS Manager.

        Args:
            tts_provider (str): Type of TTS provider ("aws", "google", etc.)
            **kwargs: Configuration parameters for the selected provider
        """
        # Validate configuration
        if tts_provider not in ["aws", "google"]:
            raise ValueError(f"Unsupported TTS provider: {tts_provider}")

        self.tts_provider = tts_provider

        try:
            config = TTSConfig(**kwargs)
            self.config = config.model_dump(exclude_unset=True)
            logger.info(f"Initialized TTS Manager with provider: {tts_provider}")
        except Exception as e:
            logger.error(f"Failed to initialize TTS Manager: {str(e)}")
            raise ValueError(f"Invalid TTS configuration: {str(e)}")

    def get_tts(self) -> Optional[BaseTTS]:
        """
        Initialize and return the selected TTS provider.

        Returns:
            BaseTTS: Configured TTS instance

        Raises:
            ValueError: If tts_provider is not supported
        """
        try:
            if self.tts_provider == "aws":
                # Additional AWS-specific validation
                if "voice_id" not in self.config:
                    logger.warning("No voice_id specified for AWS Polly, using default")

                return AWSPollyTTS(**self.config)

            elif self.tts_provider == "google":
                # Additional Google-specific validation
                if "language" not in self.config:
                    logger.warning(
                        "No language specified for Google TTS, using default"
                    )

                return GoogleTTS(**self.config)

            raise ValueError(f"Unsupported TTS provider: {self.tts_provider}")

        except Exception as e:
            logger.error(f"Failed to initialize TTS instance: {str(e)}")
            raise
