"""
Manager classes for LLM and TTS provider selection.
"""

from typing import Dict, Optional, Any
import logging
from pydantic import BaseModel, Field, validator

# Setup logging
logger = logging.getLogger(__name__)


class LLMConfig(BaseModel):
    """Configuration model for LLM providers."""

    provider_type: str = Field(..., description="Type of LLM provider")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    model_name: Optional[str] = None
    temperature: float = Field(0.2, ge=0.0, le=1.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_output_tokens: int = Field(4096, gt=0)
    streaming: bool = False

    @validator("provider_type")
    def validate_provider_type(cls, v):
        if v not in ["gemini", "openai"]:
            raise ValueError(f"Unsupported LLM provider: {v}")
        return v


class TTSConfig(BaseModel):
    """Configuration model for TTS providers."""

    provider_type: str = Field(..., description="Type of TTS provider")
    voice_id: Optional[str] = None
    language: Optional[str] = None
    region_name: Optional[str] = None
    engine: str = Field("neural", pattern="^(standard|neural)$")
    temp_dir: str = "temp"

    @validator("provider_type")
    def validate_provider_type(cls, v):
        if v not in ["aws", "google"]:
            raise ValueError(f"Unsupported TTS provider: {v}")
        return v

    @validator("region_name")
    def validate_region(cls, v, values):
        if values.get("provider_type") == "aws" and not v:
            raise ValueError("region_name is required for AWS Polly")
        return v


from .llm import GeminiLLM
from .tts import AWSPollyTTS, GoogleTTS
from .base import BaseLLM, BaseTTS


class LLMManager:
    """
    Manager class for selecting and configuring LLM providers.
    """

    def __init__(self, provider_type: str, **kwargs: Dict[str, Any]):
        """
        Initialize LLM Manager.

        Args:
            provider_type (str): Type of LLM provider ("gemini", "openai", etc.)
            **kwargs: Configuration parameters for the selected provider
        """
        try:
            # Validate configuration
            config = LLMConfig(provider_type=provider_type, **kwargs)
            self.provider_type = config.provider_type
            self.config = config.dict(exclude_unset=True)
            logger.info(f"Initialized LLM Manager with provider: {provider_type}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM Manager: {str(e)}")
            raise ValueError(f"Invalid LLM configuration: {str(e)}")

    def get_llm(self) -> Optional[BaseLLM]:
        """
        Initialize and return the selected LLM provider.

        Returns:
            BaseLLM: Configured LLM instance

        Raises:
            ValueError: If provider_type is not supported
        """
        try:
            if self.provider_type == "gemini":
                return GeminiLLM(**self.config)
            # Add support for other providers here
            raise ValueError(f"Unsupported LLM provider: {self.provider_type}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM instance: {str(e)}")
            raise


class TTSManager:
    """
    Manager class for selecting and configuring TTS providers.
    """

    def __init__(self, provider_type: str, **kwargs: Dict[str, Any]):
        """
        Initialize TTS Manager.

        Args:
            provider_type (str): Type of TTS provider ("aws", "google", etc.)
            **kwargs: Configuration parameters for the selected provider
        """
        try:
            # Validate configuration
            config = TTSConfig(provider_type=provider_type, **kwargs)
            self.provider_type = config.provider_type
            self.config = config.dict(exclude_unset=True)
            logger.info(f"Initialized TTS Manager with provider: {provider_type}")
        except Exception as e:
            logger.error(f"Failed to initialize TTS Manager: {str(e)}")
            raise ValueError(f"Invalid TTS configuration: {str(e)}")

    def get_tts(self) -> Optional[BaseTTS]:
        """
        Initialize and return the selected TTS provider.

        Returns:
            BaseTTS: Configured TTS instance

        Raises:
            ValueError: If provider_type is not supported
        """
        try:
            if self.provider_type == "aws":
                # Additional AWS-specific validation
                if "voice_id" not in self.config:
                    logger.warning("No voice_id specified for AWS Polly, using default")

                return AWSPollyTTS(**self.config)

            elif self.provider_type == "google":
                # Additional Google-specific validation
                if "language" not in self.config:
                    logger.warning(
                        "No language specified for Google TTS, using default"
                    )

                return GoogleTTS(**self.config)

            raise ValueError(f"Unsupported TTS provider: {self.provider_type}")

        except Exception as e:
            logger.error(f"Failed to initialize TTS instance: {str(e)}")
            raise
