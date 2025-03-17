"""
Advanced example demonstrating comprehensive usage of pdf2podcast library.

This example shows:
- PDF processing with text cleaning
- LLM configuration for optimal script generation
- Advanced TTS settings
- Error handling and logging
"""

import os
from pathlib import Path
import logging
from dotenv import load_dotenv

from pdf2podcast import PodcastGenerator, SimplePDFProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Load and validate environment variables."""
    load_dotenv()

    required_vars = {
        "GENAI_API_KEY": "Google API key for Gemini",
        "AWS_ACCESS_KEY_ID": "AWS access key for Polly",
        "AWS_SECRET_ACCESS_KEY": "AWS secret key for Polly",
    }

    missing = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing.append(f"{var} ({description})")

    if missing:
        raise ValueError(
            "Missing required environment variables:\n"
            + "\n".join(f"- {var}" for var in missing)
        )


def main():
    try:
        # Ensure environment is properly configured
        setup_environment()

        # Initialize PDF processor with custom settings
        pdf_processor = SimplePDFProcessor(
            max_chars_per_chunk=8000  # Increased chunk size for better context
        )

        # Create podcast generator with detailed configuration
        generator = PodcastGenerator(
            rag_system=pdf_processor,
            llm_type="gemini",
            tts_type="aws",
            llm_config={
                "api_key": os.getenv("GENAI_API_KEY"),
                "model_name": "gemini-1.5-flash",
                "temperature": 0.3,  # Lower temperature for more focused output
                "top_p": 0.9,
                "max_output_tokens": 8192,  # Increased for longer scripts
                "streaming": True,  # Enable streaming for better performance
            },
            tts_config={
                "voice_id": "Matthew",
                "region_name": "us-west-2",
                "engine": "neural",  # Use neural engine for better quality
            }
        )

        # Ensure output directory exists
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        # Generate podcast with advanced settings
        result = generator.generate(
            pdf_path="sample.pdf",
            output_path=str(output_dir / "podcast.mp3"),
            complexity="advanced",  # Generate detailed, technical content
            voice_id="Matthew",  # Override default voice
            # Additional parameters for fine-tuning
            temperature=0.4,  # Override LLM temperature for this generation
        )

        # Log results
        logger.info("✅ Podcast generated successfully!")
        logger.info(f"📄 Input PDF: sample.pdf")
        logger.info(f"📝 Script length: {len(result['script'])} characters")
        logger.info(f"🎧 Audio file: {result['audio']['path']}")
        logger.info(f"📊 Audio size: {result['audio']['size']} bytes")

        # Save the generated script for reference
        script_path = output_dir / "script.txt"
        script_path.write_text(result["script"])
        logger.info(f"💾 Script saved to: {script_path}")

    except ValueError as e:
        logger.error(f"❌ Configuration error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Processing error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
