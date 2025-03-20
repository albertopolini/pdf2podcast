"""
Simple example demonstrating basic usage of pdf2podcast library.
"""

import os
from dotenv import load_dotenv
from pdf2podcast import PodcastGenerator, SimplePDFProcessor


def main():
    # Load environment variables
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set GENAI_API_KEY environment variable")

    # Initialize PDF processor
    pdf_processor = SimplePDFProcessor()

    # Create podcast generator with basic configuration
    generator = PodcastGenerator(
        rag_system=pdf_processor,
        llm_provider="gemini",  # Specify LLM provider
        tts_provider="google",  # Specify TTS provider
        llm_config={"api_key": api_key, "max_output_tokens": 4096, "temperature": 0.2},
        tts_config={"language": "en", "tld": "com", "slow": False},
    )

    try:
        # Generate podcast
        result = generator.generate(
            pdf_path="sample.pdf",
            output_path="output.mp3",
            complexity="intermediate",
            audience="general",  # Specify target audience
        )

        # Show results
        print("✅ Podcast generated successfully!")
        print(f"📝 Script length: {len(result['script'])} characters")
        print(f"🎧 Audio file: {result['audio']['path']}")
        print(f"📊 Audio size: {result['audio']['size']} bytes")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
