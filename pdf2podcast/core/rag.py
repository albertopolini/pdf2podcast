"""
Retrieval Augmented Generation (RAG) implementations for pdf2podcast.
"""

from typing import List, Optional, Dict
import fitz  # PyMuPDF
from .base import BaseRAG


class AdvancedPDFProcessor(BaseRAG):
    """
    Advanced PDF text extraction implementation.

    This class provides enhanced PDF text extraction functionality using PyMuPDF.
    It processes PDF documents and returns their text content in a format
    suitable for podcast script generation, with support for various PDF structures
    and better text extraction capabilities.
    """

    def __init__(
        self,
        max_chars_per_chunk: int = 4000,
        extract_images: bool = False,
        metadata: bool = True,
        chunker=None,
        retriever=None,
    ):
        """
        Initialize PDF processor.

        Args:
            max_chars_per_chunk (int): Maximum characters per text chunk (default: 4000)
            extract_images (bool): Whether to extract image captions (default: False)
            metadata (bool): Whether to include document metadata (default: True)
            chunker (Optional[BaseChunker]): Custom text chunker (default: SimpleChunker)
            retriever (Optional[BaseRetriever]): Custom text retriever
        """
        from .processing import SimpleChunker

        self.max_chars_per_chunk = max_chars_per_chunk
        self.extract_images = extract_images
        self.include_metadata = metadata
        self.chunker = chunker or SimpleChunker()
        self.retriever = retriever

    def process_document(self, pdf_path: str) -> str:
        """
        Process a PDF document and extract its text content.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            str: Extracted and processed text from the PDF

        Raises:
            Exception: If PDF processing fails
        """
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)

            # Extract metadata if enabled
            content_parts = []
            if self.include_metadata:
                metadata = self._extract_metadata(doc)
                if metadata:
                    content_parts.append(metadata)

            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text
                text = page.get_text("text")
                if text:
                    # Clean up text
                    text = self._clean_text(text)
                    content_parts.append(text)

                # Extract image captions if enabled
                if self.extract_images:
                    image_text = self._extract_image_captions(page)
                    if image_text:
                        content_parts.append(image_text)

            # Join all content
            full_text = "\n\n".join(content_parts)

            # Instead of truncating, use chunker to split text
            chunks = self.chunker.chunk_text(full_text, self.max_chars_per_chunk)

            # If retriever is available, add chunks to it
            if self.retriever:
                self.retriever.add_texts(chunks)

            # Join chunks back together (they'll be retrieved later if needed)
            return "\n\n".join(chunks)

        except Exception as e:
            raise Exception(f"Failed to process PDF: {str(e)}")

    def _extract_metadata(self, doc: fitz.Document) -> Optional[str]:
        """
        Extract relevant metadata from the PDF document.

        Args:
            doc (fitz.Document): The PDF document

        Returns:
            Optional[str]: Formatted metadata string or None if no metadata found
        """
        metadata = doc.metadata
        if not metadata:
            return None

        relevant_fields = ["title", "author", "subject", "keywords"]
        meta_parts = []

        for field in relevant_fields:
            if metadata.get(field):
                meta_parts.append(f"{field.title()}: {metadata[field]}")

        if meta_parts:
            return "Document Information:\n" + "\n".join(meta_parts)

        return None

    def _extract_image_captions(self, page: fitz.Page) -> Optional[str]:
        """
        Extract image captions from a page.

        Args:
            page (fitz.Page): The PDF page to process

        Returns:
            Optional[str]: Extracted image captions or None if no images found
        """
        try:
            image_text = []

            # Get blocks that might contain images and their captions
            page_dict = page.get_text("dict")
            if not page_dict or "blocks" not in page_dict:
                return None

            for block in page_dict["blocks"]:
                try:
                    # Check if block contains an image
                    if block.get("type") == 1:  # type 1 indicates image block
                        # Get the bounding box coordinates
                        coords = block.get("bbox")
                        if not coords or len(coords) != 4:
                            continue

                        # Create rectangle and get surrounding text
                        rect = fitz.Rect(coords)
                        rect.x0 -= 20  # Expand left
                        rect.y0 -= 20  # Expand top
                        rect.x1 += 20  # Expand right
                        rect.y1 += 20  # Expand bottom

                        # Get text within the expanded rectangle
                        nearby_text = page.get_text("text", clip=rect).strip()
                        if nearby_text:
                            image_text.append(f"Image Caption: {nearby_text}")

                except Exception:
                    continue  # Skip problematic blocks

            return "\n".join(image_text) if image_text else None

        except Exception:
            return None  # Return None if anything goes wrong

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing unnecessary whitespace and artifacts.

        Args:
            text (str): Raw text to clean

        Returns:
            str: Cleaned text
        """
        # Remove multiple spaces
        text = " ".join(text.split())

        # Remove form feed characters
        text = text.replace("\f", "")

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove multiple newlines
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")

        # Remove common PDF artifacts
        text = text.replace("•", "")  # Remove bullet points
        text = text.replace("", "")  # Remove zero-width spaces

        return text.strip()

    def _truncate_to_complete_sentence(self, text: str, max_length: int) -> str:
        """
        Truncate text to specified length while keeping complete sentences.

        Args:
            text (str): Text to truncate
            max_length (int): Maximum length of the truncated text

        Returns:
            str: Truncated text ending with a complete sentence
        """
        if len(text) <= max_length:
            return text

        # Find the last sentence boundary before max_length
        truncated = text[:max_length]

        # Look for common sentence endings
        for end in [".", "!", "?"]:
            last_period = truncated.rfind(end)
            if last_period != -1:
                return text[: last_period + 1]

        # If no sentence boundary found, look for the last complete word
        last_space = truncated.rfind(" ")
        if last_space != -1:
            return text[:last_space]

        return truncated
