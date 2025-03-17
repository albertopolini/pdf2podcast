"""
Text processing implementations including chunking and semantic retrieval.
"""

from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from .base import BaseChunker, BaseRetriever


class SimpleChunker(BaseChunker):
    """
    Basic text chunking implementation that splits text by paragraphs.
    """

    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Split text into chunks based on paragraphs while respecting size limits.

        Args:
            text (str): Text to be chunked
            chunk_size (int): Maximum size of each chunk in characters

        Returns:
            List[str]: List of text chunks
        """
        chunks = []
        paragraphs = text.split("\n")
        current_chunk = ""

        for para in paragraphs:
            # Skip empty paragraphs
            if not para.strip():
                continue

            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) > chunk_size:
                if current_chunk:  # Save current chunk if not empty
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n" + para if current_chunk else para

        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks


class SemanticRetriever(BaseRetriever):
    """
    Semantic text retrieval using FAISS and sentence transformers.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: Optional[int] = None,
    ):
        """
        Initialize the semantic retriever.

        Args:
            model_name (str): Name of the sentence transformer model to use
            dimension (Optional[int]): Embedding dimension (if known)
        """
        self.model = SentenceTransformer(model_name)
        self.texts: List[str] = []

        # Initialize FAISS index
        self.dimension = dimension or self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts."""
        return self.model.encode(texts, convert_to_numpy=True)

    def add_texts(self, texts: List[str]) -> None:
        """
        Add texts to the retrieval system.

        Args:
            texts (List[str]): List of text chunks to be indexed
        """
        if not texts:
            return

        # Store texts
        self.texts.extend(texts)

        # Get embeddings and add to index
        embeddings = self._get_embeddings(texts)
        self.index.add(embeddings.astype(np.float32))

    def get_relevant_chunks(self, query: str, k: int = 5) -> List[str]:
        """
        Retrieve most relevant text chunks for a query.

        Args:
            query (str): Query text to find relevant chunks for
            k (int): Number of chunks to retrieve (default: 3)

        Returns:
            List[str]: List of relevant text chunks

        Note:
            If k is greater than the number of stored chunks,
            all chunks will be returned in order of relevance.
        """
        if not self.texts:  # No texts indexed yet
            return []

        # Convert k to valid range
        k = min(k, len(self.texts))

        # Get query embedding
        query_embedding = self._get_embeddings([query])

        # Search for similar vectors
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)

        # Return corresponding texts
        return [self.texts[idx] for idx in indices[0]]
