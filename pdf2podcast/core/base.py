"""
Base abstract classes for the pdf2podcast library components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BasePromptBuilder(ABC):
    """Base class for building prompts."""

    @abstractmethod
    def build_prompt(self, text: str, **kwargs) -> str:
        """
        Build a prompt for content generation.

        Args:
            text (str): Source text
            **kwargs: Additional prompt parameters

        Returns:
            str: Formatted prompt
        """
        pass

    @abstractmethod
    def build_expand_prompt(self, text: str, **kwargs) -> str:
        """
        Build a prompt for content expansion.

        Args:
            text (str): Current content
            **kwargs: Additional prompt parameters

        Returns:
            str: Formatted expansion prompt
        """
        pass


class BaseRAG(ABC):
    """Base class for RAG (Retrieval Augmented Generation) implementations."""

    @abstractmethod
    def process_document(self, pdf_path: str) -> str:
        """
        Process a PDF document and extract relevant text content.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            str: Extracted and processed text from the PDF
        """
        pass


class BaseChunker(ABC):
    """Base class for text chunking implementations."""

    @abstractmethod
    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Split text into manageable chunks.

        Args:
            text (str): Text to be chunked
            chunk_size (int): Maximum size of each chunk in characters

        Returns:
            List[str]: List of text chunks
        """
        pass


class BaseRetriever(ABC):
    """Base class for semantic text retrieval implementations."""

    @abstractmethod
    def add_texts(self, texts: List[str]) -> None:
        """
        Add texts to the retrieval system.

        Args:
            texts (List[str]): List of text chunks to be indexed
        """
        pass

    @abstractmethod
    def get_relevant_chunks(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve most relevant text chunks for a query.

        Args:
            query (str): Query text to find relevant chunks for
            k (int): Number of chunks to retrieve (default: 3)

        Returns:
            List[str]: List of relevant text chunks
        """
        pass


class BaseLLM(ABC):
    """Base class for Large Language Model implementations."""

    def __init__(self, prompt_builder: Optional[BasePromptBuilder] = None):
        """
        Initialize LLM with optional prompt builder.

        Args:
            prompt_builder (Optional[BasePromptBuilder]): Custom prompt builder
        """
        self.prompt_builder = prompt_builder

    @abstractmethod
    def generate_podcast_script(
        self, text: str, complexity: str = "intermediate", **kwargs: Dict[str, Any]
    ) -> str:
        """
        Generate a podcast script from input text.

        Args:
            text (str): Input text to convert into a podcast script
            complexity (str): Desired complexity level of the output
                            ("simple", "intermediate", "advanced")
            **kwargs: Additional model-specific parameters

        Returns:
            str: Generated podcast script
        """
        pass


class BaseTTS(ABC):
    """Base class for Text-to-Speech implementations."""

    @abstractmethod
    def generate_audio(
        self,
        text: str,
        output_path: str,
        voice_id: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert text to speech and save as audio file.

        Args:
            text (str): Text to convert to speech
            output_path (str): Path where to save the audio file
            voice_id (Optional[str]): ID of the voice to use
            **kwargs: Additional TTS-specific parameters

        Returns:
            Dict[str, Any]: Dictionary containing audio metadata
                          (e.g., {'duration': float, 'size': int})
        """
        pass


class BasePodcastGenerator:
    """Base class for podcast generation orchestration."""

    def __init__(
        self,
        rag_system: BaseRAG,
        llm_provider: str,
        tts_provider: str,
        llm_config: Optional[Dict[str, Any]] = None,
        tts_config: Optional[Dict[str, Any]] = None,
        chunker: Optional[BaseChunker] = None,
        retriever: Optional[BaseRetriever] = None,
        k: int = 3,
    ):
        """
        Initialize podcast generator with required components.

        Args:
            rag_system (BaseRAG): System for PDF text extraction
            llm_provider (str): Type of LLM to use ("gemini", "openai", etc.)
            tts_provider (str): Type of TTS to use ("aws", "google", etc.)
            llm_config (Optional[Dict[str, Any]]): Configuration for LLM
            tts_config (Optional[Dict[str, Any]]): Configuration for TTS
            chunker (Optional[BaseChunker]): System for text chunking
            retriever (Optional[BaseRetriever]): System for semantic retrieval
            k (int): Number of chunks to retrieve for a query (default: 3)
        """
        from .managers import LLMManager, TTSManager

        self.rag = rag_system

        # Initialize models using managers
        llm_manager = LLMManager(llm_provider, **(llm_config or {}))
        tts_manager = TTSManager(tts_provider, **(tts_config or {}))

        self.llm = llm_manager.get_llm()
        self.tts = tts_manager.get_tts()
        self.chunker = chunker
        self.retriever = retriever
        self.k = k

    def generate(
        self,
        pdf_path: str,
        output_path: str,
        complexity: str = "intermediate",
        voice_id: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a podcast from a PDF document.

        Args:
            pdf_path (str): Path to the input PDF file
            output_path (str): Path where to save the output audio file
            complexity (str): Desired complexity of the podcast script
            voice_id (Optional[str]): ID of the voice to use for TTS
            query (Optional[str]): Query for semantic retrieval of relevant chunks
            **kwargs: Additional parameters for RAG, LLM, or TTS systems

        Returns:
            Dict[str, Any]: Dictionary containing generation results and metadata
        """
        # Extract text from PDF
        text = self.rag.process_document(pdf_path)

        # Process with chunking and retrieval if available
        if self.chunker and self.retriever:
            chunks = self.chunker.chunk_text(text)
            self.retriever.add_texts(chunks)

            if query:
                # Use retrieved chunks if query provided
                relevant_chunks = self.retriever.get_relevant_chunks(query, k=self.k)
                text = "\n\n".join(relevant_chunks)

        # Generate podcast script
        script = self.llm.generate_podcast_script(
            text=text, complexity=complexity, **kwargs
        )

        # Convert script to audio
        audio_result = self.tts.generate_audio(
            text=script, output_path=output_path, voice_id=voice_id, **kwargs
        )

        return {"script": script, "audio": audio_result}
