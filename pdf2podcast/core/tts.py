"""
Text-to-Speech (TTS) implementations for pdf2podcast.
"""

import os
import time
from typing import Dict, Any, Optional, List, Callable
from contextlib import closing
import tempfile
import logging
from functools import wraps

# AWS Polly
import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Google TTS
from gtts import gTTS
from gtts.tts import gTTSError

# Audio processing
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# Setup logging
logger = logging.getLogger(__name__)


def retry_on_exception(
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Retry decorator with exponential backoff.

    Args:
        retries (int): Maximum number of retries
        delay (float): Initial delay between retries in seconds
        backoff (float): Backoff multiplier
        exceptions (tuple): Exceptions to catch
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_delay = delay
            last_exception = None

            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if i < retries - 1:
                        logger.warning(
                            f"Attempt {i + 1}/{retries} failed: {str(e)}. "
                            f"Retrying in {retry_delay:.1f}s..."
                        )
                        time.sleep(retry_delay)
                        retry_delay *= backoff
                    else:
                        logger.error(f"All {retries} attempts failed.")

            raise last_exception

        return wrapper

    return decorator


from .base import BaseTTS


def validate_audio_file(file_path: str) -> bool:
    """
    Validate that an audio file is properly formatted MP3.

    Args:
        file_path (str): Path to the audio file

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        audio = AudioSegment.from_mp3(file_path)
        return len(audio) > 0
    except (CouldntDecodeError, OSError):
        return False
    except Exception as e:
        logger.error(f"Unexpected error validating audio: {str(e)}")
        return False


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


def merge_audio_files(files: List[str], output_file: str) -> bool:
    """
    Merge multiple MP3 files into one.

    Args:
        files (List[str]): List of MP3 file paths
        output_file (str): Path for the merged file
    """
    try:
        combined = AudioSegment.empty()
        valid_files = []

        # Validate and load audio files
        for file in files:
            if validate_audio_file(file):
                audio = AudioSegment.from_mp3(file)
                combined += audio
                valid_files.append(file)
            else:
                logger.error(f"Invalid audio file: {file}")

        if not valid_files:
            raise ValueError("No valid audio files to merge")

        # Export combined audio
        combined.export(output_file, format="mp3")

        # Clean up temporary files only after successful export
        for file in valid_files:
            try:
                os.remove(file)
            except OSError as e:
                logger.warning(f"Failed to remove temporary file {file}: {str(e)}")

        return True

    except Exception as e:
        logger.error(f"Error merging audio files: {str(e)}")
        # Attempt cleanup on failure
        for file in files:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except OSError:
                pass
        return False


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

    @retry_on_exception(retries=3, delay=1.0, exceptions=(BotoCoreError, ClientError))
    def _generate_chunk(
        self, text: str, output_path: str, voice_id: Optional[str] = None
    ) -> bool:
        """
        Generate audio for a single text chunk with retry mechanism.

        Args:
            text (str): Text to convert
            output_path (str): Where to save the audio
            voice_id (Optional[str]): Override default voice

        Returns:
            bool: True if successful

        Raises:
            BotoCoreError: For AWS SDK related errors
            ClientError: For AWS API related errors
        """
        # Use provided voice_id or default
        voice_id = voice_id or self.voice_id

        try:
            response = self.polly.synthesize_speech(
                Text=text, OutputFormat="mp3", VoiceId=voice_id, Engine=self.engine
            )

            if "AudioStream" not in response:
                logger.error("No AudioStream in Polly response")
                return False

            with closing(response["AudioStream"]) as stream:
                with open(output_path, "wb") as file:
                    file.write(stream.read())

            # Validate generated audio file
            if not validate_audio_file(output_path):
                logger.error(f"Generated audio file {output_path} is invalid")
                return False

            return True

        except (BotoCoreError, ClientError) as e:
            logger.error(f"AWS Polly error: {str(e)}")
            raise  # Will be caught by retry decorator
        except Exception as e:
            logger.error(f"Unexpected error in audio generation: {str(e)}")
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

    @retry_on_exception(retries=3, delay=2.0, exceptions=(gTTSError,))
    def _generate_chunk(
        self, text: str, output_path: str, language: Optional[str] = None
    ) -> bool:
        """
        Generate audio for a single text chunk using gTTS with retry mechanism.

        Args:
            text (str): Text to convert
            output_path (str): Where to save the audio
            language (Optional[str]): Override default language

        Returns:
            bool: True if successful

        Raises:
            gTTSError: For Google TTS specific errors
        """
        try:
            # Use provided language or default
            lang = language or self.language

            # Create gTTS object and save audio
            tts = gTTS(text=text, lang=lang, slow=self.slow, tld=self.tld)
            tts.save(output_path)

            # Validate generated audio file
            if not validate_audio_file(output_path):
                logger.error(f"Generated audio file {output_path} is invalid")
                return False

            return True

        except gTTSError as e:
            logger.error(f"Google TTS error: {str(e)}")
            raise  # Will be caught by retry decorator
        except Exception as e:
            logger.error(f"Unexpected error in audio generation: {str(e)}")
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
