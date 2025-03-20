# PDF2Podcast 🎙️

Transform PDF documents into engaging, narrative-driven audio content using state-of-the-art AI technology.

[![PyPI version](https://badge.fury.io/py/pdf2podcast.svg)](https://badge.fury.io/py/pdf2podcast)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

PDF2Podcast leverages advanced RAG (Retrieval Augmented Generation) technology, LLMs, and TTS capabilities to convert technical documents into professional, narrative-style podcasts. It intelligently processes PDF content, maintains context across sections, and generates natural-sounding audio output.

## Key Features

🔍 **Smart Document Processing**
- Advanced PDF text extraction with support for complex layouts
- Image caption extraction and metadata integration
- Intelligent chunking with semantic context preservation

🧠 **AI-Powered Content Generation**
- Context-aware content transformation using RAG technology
- Customizable complexity levels and audience targeting
- Support for narrative-style content generation

🗣️ **Professional Audio Output**
- High-quality text-to-speech synthesis
- Multiple voice provider options
- Natural pacing and pronunciation

⚙️ **Flexible Configuration**
- Modular architecture for easy extension
- Configurable LLM and TTS providers
- Custom prompt builder support

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Advanced Usage](#advanced-usage)
  - [Custom Prompt Builders](#custom-prompt-builders)
  - [Provider Configuration](#provider-configuration)
- [Configuration Reference](#configuration-reference)
- [Environment Setup](#environment-setup)
- [Error Handling](#error-handling)
- [License](#license)

## Installation

Install PDF2Podcast using pip:

```bash
pip install pdf2podcast
```

## Quick Start

Here's a basic example to get you started with PDF2Podcast:

```python
from pdf2podcast import PodcastGenerator, SimplePDFProcessor
import os
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

# Initialize the PDF processor
# This component handles document reading, content extraction, and chunking
pdf_processor = SimplePDFProcessor()

# Create a podcast generator instance with basic configuration
# - llm_provider: Specifies which LLM service to use (currently supports "gemini")
# - tts_provider: Specifies which TTS service to use ("google" or "aws")
# - llm_config: Configuration for the LLM service
# - tts_config: Configuration for the TTS service
generator = PodcastGenerator(
    rag_system=pdf_processor,
    llm_provider="gemini",         # Using Google's Gemini model
    tts_provider="google",         # Using Google's TTS service
    llm_config={
        "api_key": os.getenv("GENAI_API_KEY"),  # API key from environment variables
        "max_output_tokens": 4096,  # Maximum length of generated content
        "temperature": 0.2         # Controls creativity vs determinism (0.0-1.0)
    },
    tts_config={
        "language": "en",          # Output language
        "tld": "com",             # TLD for accent selection
        "slow": False             # Normal speech rate
    }
)

# Generate the podcast
# - complexity: Controls content complexity ("simple", "intermediate", "advanced")
# - audience: Target audience type ("general", "enthusiasts", "professionals", "experts")
result = generator.generate(
    pdf_path="sample.pdf",         # Input PDF file
    output_path="output.mp3",      # Output audio file
    complexity="intermediate",      # Moderate technical depth
    audience="general"             # General audience targeting
)

# The result contains:
# - script: The generated narrative script
# - audio: Dictionary with audio file details (path, size)
print(f"Generated audio file: {result['audio']['path']}")
print(f"Script length: {len(result['script'])} characters")
```

This basic example demonstrates:
1. Setting up the PDF processor for content extraction
2. Configuring the podcast generator with LLM and TTS providers
3. Generating a podcast with customized complexity and audience targeting
4. Accessing the generated content and audio file

## Advanced Usage

### Custom Prompt Builders

PDF2Podcast supports custom prompt builders to control how content is transformed into narrative form. The library includes a `StorytellingPromptBuilder` that creates engaging, story-like narratives from technical content:

```python
from pdf2podcast import PodcastGenerator, SimplePDFProcessor
from pdf2podcast.examples.custom_prompts import StorytellingPromptBuilder

# Initialize PDF processor with advanced settings
pdf_processor = SimplePDFProcessor(
    max_chars_per_chunk=6000,  # Larger chunks for better context
    extract_images=True,       # Include image captions in processing
    metadata=True             # Include document metadata
)

# Create generator with storytelling configuration
generator = PodcastGenerator(
    rag_system=pdf_processor,
    llm_provider="gemini",
    tts_provider="google",
    llm_config={
        "api_key": os.getenv("GENAI_API_KEY"),
        "model_name": "gemini-1.5-flash",    # Specific model version
        "max_output_tokens": 8000,           # Longer output for stories
        "temperature": 0.3,                  # Slightly more creative
        "prompt_builder": StorytellingPromptBuilder()  # Enable storytelling mode
    },
    tts_config={
        "language": "en",
        "tld": "com",
        "slow": False
    }
)

# Generate a narrative-style podcast
result = generator.generate(
    pdf_path="sample.pdf",
    output_path="output.mp3",
    complexity="advanced",        # Detailed technical content
    audience="enthusiasts",      # For technically interested listeners
    query="Explain the main concepts and their practical applications"
)
```

The StorytellingPromptBuilder transforms content by:
- Creating engaging hooks and openings
- Building narrative tension around technical concepts
- Presenting solutions as revelations
- Maintaining technical accuracy while being engaging
- Adapting language and examples to the target audience

### Provider Configuration

#### LLM Provider Settings

The library currently supports Google's Gemini as the LLM provider:

```python
# Full configuration options for Gemini
generator = PodcastGenerator(
    llm_provider="gemini",  # Specify Gemini as provider
    llm_config={
        "api_key": os.getenv("GENAI_API_KEY"),
        "model_name": "gemini-1.5-flash",  # Model version (optional)
        "max_output_tokens": 4096,         # Maximum response length
        "temperature": 0.2,                # Creativity control (0.0-1.0)
        "top_p": 0.9,                     # Nucleus sampling parameter
        "streaming": False                 # Enable/disable streaming
    },
    ...
    
)
```

#### TTS Provider Settings

Two TTS providers are supported with their own configuration options:

**Google TTS**
```python
# Google TTS configuration
generator = PodcastGenerator(
    tts_provider="google",  # Use Google's TTS service
    tts_config={
        "language": "en",    # Language code
        "tld": "com",       # Top-level domain for accent
        "slow": False,      # Speech rate
    },
    ...
)
```

**AWS Polly**
```python
# AWS Polly configuration
generator = PodcastGenerator(
    tts_provider="aws",  # Use AWS Polly
    tts_config={
        "voice_id": "Joanna",          # Voice selection
        "region_name": "eu-central-1", # AWS region
        "engine": "neural",            # Neural TTS engine
        # Additional options:
        # "sample_rate": 22050,        # Audio sample rate
        # "audio_format": "mp3"        # Output format
    },
    ...
)
```

## Configuration Reference

### Complexity Levels

| Level | Description | Best For |
|-------|-------------|-----------|
| simple | Basic terms, clear explanations | General audience, introductory content |
| intermediate | Balanced technical depth | Students, professionals |
| advanced | Full technical detail | Experts, technical documentation |

### Audience Types

| Type | Description | Content Adaptation |
|------|-------------|-------------------|
| general | No technical background | Focus on practical understanding |
| enthusiasts | Interest-driven knowledge | Hobby and DIY applications |
| professionals | Working knowledge | Industry applications |
| experts | Deep domain knowledge | Advanced concepts |

## Environment Setup

Required environment variables:

```env
# Gemini API Configuration
GENAI_API_KEY=your_gemini_api_key

# AWS Polly Configuration (if using)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
```

## Error Handling

PDF2Podcast provides comprehensive error handling with specific exceptions:

```python
try:
    # Attempt to generate podcast
    result = generator.generate(
        pdf_path="sample.pdf",
        output_path="output.mp3"
    )
except ValueError as e:
    # Handle configuration errors
    print(f"Configuration error: {str(e)}")
    # Example: Invalid API keys, unsupported providers
except FileNotFoundError as e:
    # Handle file access errors
    print(f"File error: {str(e)}")
    # Example: PDF not found, output directory issues
except Exception as e:
    # Handle other processing errors
    print(f"Processing error: {str(e)}")
    # Example: Network issues, service failures
```

Common errors and solutions:
- Configuration errors: Check API keys and provider settings
- File errors: Verify file paths and permissions
- Processing errors: Check network connection and service status

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
