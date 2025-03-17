"""
Text-to-Speech (TTS) implementations for pdf2podcast.
"""

import os
from typing import Dict, Any, Optional, List
from contextlib import closing
import tempfile

# AWS Polly
import boto3

# Google TTS
from gtts import gTTS

# Audio processing
from pydub import AudioSegment

from .base import BaseTTS


def split_text(text: str, max_length: int = 3000) -> List[str]:
    """
    Split text into chunks that are safe for TTS processing.

    Args:
        text (str): Text to split
        max_length (int): Maximum length per chunk

    Returns:
        List[str]: List of text chunks
    """
    chunks = []
    sentences = text.split(". ")
    current_chunk = ""

    for sentence in sentences:
        if not sentence.strip():
            continue

        # Add period back if it was removed by split
        sentence = sentence.strip() + ". "

        if len(current_chunk) + len(sentence) > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def merge_audio_files(files: List[str], output_file: str) -> None:
    """
    Merge multiple MP3 files into one.

    Args:
        files (List[str]): List of MP3 file paths
        output_file (str): Path for the merged file
    """
    combined = AudioSegment.empty()
    for file in files:
        audio = AudioSegment.from_mp3(file)
        combined += audio
        # Clean up temporary file
        os.remove(file)

    combined.export(output_file, format="mp3")


class AWSPollyTTS(BaseTTS):
    """
    AWS Polly-based Text-to-Speech implementation.

    This class provides TTS functionality using Amazon's Polly service.
    Requires AWS credentials to be configured through environment variables
    or AWS configuration files.
    """

    def __init__(
        self,
        voice_id: str = "Joanna",
        region_name: str = "eu-central-1",
        engine: str = "neural",
        temp_dir: str = "temp",
    ):
        """
        Initialize AWS Polly TTS service.

        Args:
            voice_id (str): ID of the voice to use (default: "Joanna")
            region_name (str): AWS region for Polly service (default: "eu-central-1")
            engine (str): Polly engine type - "standard" or "neural" (default: "neural")
            temp_dir (str): Directory for temporary files (default: "temp")
        """
        self.polly = boto3.client("polly", region_name=region_name)
        self.voice_id = voice_id
        self.engine = engine
        self.temp_dir = temp_dir

        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)

    def _generate_chunk(
        self, text: str, output_path: str, voice_id: Optional[str] = None
    ) -> bool:
        """
        Generate audio for a single text chunk.

        Args:
            text (str): Text to convert
            output_path (str): Where to save the audio
            voice_id (Optional[str]): Override default voice

        Returns:
            bool: True if successful
        """
        try:
            # Use provided voice_id or default
            voice_id = voice_id or self.voice_id

            response = self.polly.synthesize_speech(
                Text=text, OutputFormat="mp3", VoiceId=voice_id, Engine=self.engine
            )

            if "AudioStream" in response:
                with closing(response["AudioStream"]) as stream:
                    with open(output_path, "wb") as file:
                        file.write(stream.read())
                return True

        except Exception as e:
            print(f"Error generating chunk: {str(e)}")
            return False

        return False

    def generate_audio(
        self,
        text: str,
        output_path: str,
        voice_id: Optional[str] = None,
        max_chunk_length: int = 3000,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert text to speech and save as audio file.

        Args:
            text (str): Text to convert to speech
            output_path (str): Path where to save the audio file
            voice_id (Optional[str]): ID of the voice to use
            max_chunk_length (int): Maximum text length per chunk
            **kwargs: Additional TTS-specific parameters

        Returns:
            Dict[str, Any]: Dictionary containing audio metadata
                          (e.g., {'path': str, 'size': int})
        """
        try:
            # Split text into chunks
            chunks = split_text(text, max_chunk_length)
            chunk_files = []

            # Generate audio for each chunk
            for i, chunk in enumerate(chunks):
                chunk_path = os.path.join(self.temp_dir, f"chunk_{i}.mp3")
                if self._generate_chunk(chunk, chunk_path, voice_id):
                    chunk_files.append(chunk_path)

            if not chunk_files:
                raise Exception("No audio chunks were generated")

            # Merge chunks if there are multiple
            if len(chunk_files) > 1:
                merge_audio_files(chunk_files, output_path)
            else:
                # Just rename single chunk file
                os.rename(chunk_files[0], output_path)

            # Get file size
            size = os.path.getsize(output_path)

            return {"success": True, "path": output_path, "size": size}

        except Exception as e:
            return {"success": False, "error": str(e), "path": None, "size": 0}


class GoogleTTS(BaseTTS):
    """
    Google Text-to-Speech implementation using gTTS.

    This class provides TTS functionality using Google's TTS service through gTTS.
    No API key required, but has usage limitations and fewer voice options.
    """

    def __init__(
        self,
        language: str = "en",
        tld: str = "com",
        slow: bool = False,
        temp_dir: str = "temp",
    ):
        """
        Initialize Google TTS service.

        Args:
            language (str): Language code (default: "en")
            tld (str): Top-level domain for accent (default: "com")
            slow (bool): Slower audio output (default: False)
            temp_dir (str): Directory for temporary files (default: "temp")
        """
        self.language = language
        self.tld = tld
        self.slow = slow
        self.temp_dir = temp_dir

        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)

    def _generate_chunk(
        self, text: str, output_path: str, language: Optional[str] = None
    ) -> bool:
        """
        Generate audio for a single text chunk using gTTS.

        Args:
            text (str): Text to convert
            output_path (str): Where to save the audio
            language (Optional[str]): Override default language

        Returns:
            bool: True if successful
        """
        try:
            # Use provided language or default
            lang = language or self.language

            # Create gTTS object and save audio
            tts = gTTS(text=text, lang=lang, slow=self.slow, tld=self.tld)
            tts.save(output_path)
            return True

        except Exception as e:
            print(f"Error generating chunk: {str(e)}")
            return False

    def generate_audio(
        self,
        text: str,
        output_path: str,
        language: Optional[str] = None,
        max_chunk_length: int = 5000,  # gTTS has a different limit than Polly
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert text to speech using Google TTS and save as audio file.

        Args:
            text (str): Text to convert to speech
            output_path (str): Path where to save the audio file
            language (Optional[str]): Language code to use
            max_chunk_length (int): Maximum text length per chunk
            **kwargs: Additional TTS-specific parameters

        Returns:
            Dict[str, Any]: Dictionary containing audio metadata
                          (e.g., {'path': str, 'size': int})
        """
        try:
            # Split text into chunks
            chunks = split_text(text, max_chunk_length)
            chunk_files = []

            # Generate audio for each chunk
            for i, chunk in enumerate(chunks):
                chunk_path = os.path.join(self.temp_dir, f"chunk_{i}.mp3")
                if self._generate_chunk(chunk, chunk_path, language):
                    chunk_files.append(chunk_path)

            if not chunk_files:
                raise Exception("No audio chunks were generated")

            # Merge chunks if there are multiple
            if len(chunk_files) > 1:
                merge_audio_files(chunk_files, output_path)
            else:
                # Just rename single chunk file
                os.rename(chunk_files[0], output_path)

            # Get file size
            size = os.path.getsize(output_path)

            return {"success": True, "path": output_path, "size": size}

        except Exception as e:
            return {"success": False, "error": str(e), "path": None, "size": 0}
