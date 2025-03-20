"""
Advanced example demonstrating comprehensive usage of pdf2podcast library.

This example shows:
- Custom storytelling prompt builder usage
- Advanced PDF processing settings
- Error handling and logging
"""

import os
from pathlib import Path
import logging
from dotenv import load_dotenv

from pdf2podcast import PodcastGenerator, SimplePDFProcessor
from pdf2podcast.examples.custom_prompts import StorytellingPromptBuilder

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


def generate_storytelling_podcast():
    """Generate a podcast using storytelling style."""
    # Initialize components with storytelling configuration
    pdf_processor = SimplePDFProcessor(
        max_chars_per_chunk=6000,  # Larger chunks for better context
        extract_images=True,  # Include image captions if available
        metadata=True,  # Include document metadata
    )

    # Create generator with storytelling prompt builder
    generator = PodcastGenerator(
        rag_system=pdf_processor,
        llm_provider="gemini",
        tts_provider="google",
        llm_config={
            "api_key": os.getenv("GENAI_API_KEY"),
            "model_name": "gemini-1.5-flash",
            "max_output_tokens": 8000,
            "temperature": 0.3,
            "prompt_builder": StorytellingPromptBuilder(),  # Use storytelling style
        },
        tts_config={"language": "en", "tld": "com", "slow": False},
    )

    return generator.generate(
        pdf_path="sample.pdf",
        output_path="output_story.mp3",
        complexity="advanced",  # Higher complexity for detailed content
        audience="enthusiasts",  # Target technically interested audience
        query="Explain the main concepts and their practical applications",
    )


def save_script(script: str, filename: str):
    """Save generated script to file."""
    script_dir = Path("output")
    script_dir.mkdir(exist_ok=True)

    script_path = script_dir / filename
    script_path.write_text(script)
    logger.info(f"💾 Script saved to: {script_path}")


def main():
    try:
        # Ensure environment is properly configured
        setup_environment()

        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        # Generate storytelling version
        logger.info("Generating storytelling version...")
        result = generate_storytelling_podcast()

        # Save and log results
        save_script(result["script"], "storytelling_script.txt")
        logger.info("✅ Storytelling version complete")
        logger.info(f"📝 Script length: {len(result['script'])} characters")
        logger.info(f"🎧 Audio file: {result['audio']['path']}")
        logger.info(f"📊 Audio size: {result['audio']['size']} bytes")

    except ValueError as e:
        logger.error(f"❌ Configuration error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Processing error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
