"""
pdf2podcast - A Python library to convert PDF documents into podcasts.
"""

from .core.base import (
    BasePodcastGenerator,
    BaseRAG,
    BaseLLM,
    BaseTTS,
    BaseChunker,
    BaseRetriever,
    BasePromptBuilder,
)
from .core.rag import (
    AdvancedPDFProcessor as SimplePDFProcessor,
)  # For backward compatibility
from .core.llm import GeminiLLM
from .core.tts import AWSPollyTTS
from .core.prompts import PodcastPromptBuilder
from .core.processing import SimpleChunker, SemanticRetriever


# Main podcast generator class
class PodcastGenerator(BasePodcastGenerator):
    """
    Main class for converting PDFs to podcasts.

    This class provides a simple interface for converting PDF documents
    into audio podcasts using configurable RAG, LLM, and TTS components.
    """

    pass  # Inherits all functionality from BasePodcastGenerator


__version__ = "0.1.0"

__all__ = [
    "PodcastGenerator",
    "SimplePDFProcessor",
    "GeminiLLM",
    "AWSPollyTTS",
    "BaseRAG",
    "BaseLLM",
    "BaseTTS",
    "BaseChunker",
    "BaseRetriever",
    "SimpleChunker",
    "SemanticRetriever",
]
