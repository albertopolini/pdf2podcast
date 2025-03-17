"""
Simple example demonstrating basic usage of pdf2podcast library.
"""

from ..pdf2podcast import PodcastGenerator, SimplePDFProcessor
import os
from dotenv import load_dotenv


def main():
    # Load environment variables
    load_dotenv()

    # Get API keys from environment
    google_api_key = os.getenv("GENAI_API_KEY")
    if not google_api_key:
        raise ValueError("Please set GENAI_API_KEY environment variable")

    # Initialize components
    pdf_processor = SimplePDFProcessor()

    # Create podcast generator with configuration
    generator = PodcastGenerator(
        rag_system=pdf_processor,
        llm_type="gemini",
        tts_type="aws",
        llm_config={
            "api_key": google_api_key,
            "model_name": "gemini-1.5-flash",
            "temperature": 0.2,
        },
        tts_config={"voice_id": "Joanna", "region_name": "eu-central-1"},
    )

    # Generate podcast
    try:
        result = generator.generate(
            pdf_path="sample.pdf",  # Replace with your PDF file
            output_path="output.mp3",
            complexity="intermediate",
        )
        print("✅ Podcast generated successfully!")
        print(f"📝 Script length: {len(result['script'])} characters")
        print(f"🎧 Audio file: {result['audio']['path']}")
        print(f"📊 Audio size: {result['audio']['size']} bytes")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
