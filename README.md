# pdf2podcast

A Python library to convert PDF documents into podcasts using LLMs and Text-to-Speech.

## Installation

```bash
pip install pdf2podcast
```

## Requirements

- Python 3.8 or higher
- Google API key for Gemini LLM
- AWS credentials for Polly TTS (optional, can use Google TTS instead)

## Dependencies

This library uses several key technologies:

- **Text Processing**
  - `PyMuPDF`: Advanced PDF processing with metadata and image caption extraction
  - `sentence-transformers`: Text embeddings for semantic analysis
  - `faiss-cpu`: Fast similarity search for text chunks
  
- **Language Models**
  - `langchain-google-genai`: Integration with Google's Gemini LLM
  - `langchain-community`: Core LangChain functionality
  - `accelerate`: ML model optimization
  
- **Text-to-Speech**
  - `boto3`: AWS Polly integration for high-quality TTS
  - `ffmpeg-python`: Audio processing and manipulation
  - `gTTS`: Google Text-to-Speech alternative

- **Utils**
  - `python-dotenv`: Environment variable management
  - `pydantic`: Data validation and settings management
  - `datasets`: Data handling utilities

## Quick Start

```python
from pdf2podcast import PodcastGenerator, SimplePDFProcessor

# Initialize PDF processor with advanced features
pdf_processor = SimplePDFProcessor(
    max_chars_per_chunk=8000,    # Customize chunk size
    extract_images=True,         # Include image captions
    metadata=True               # Include document metadata
)

# Create podcast generator with configuration
generator = PodcastGenerator(
    rag_system=pdf_processor,
    llm_type="gemini",      # Specify LLM provider
    tts_type="aws",         # Specify TTS provider
    llm_config={
        "api_key": "your_google_api_key",
        "model_name": "gemini-1.5-flash",
        "temperature": 0.2
    },
    tts_config={
        "voice_id": "Joanna",
        "region_name": "us-west-2"
    }
)

# Generate podcast
result = generator.generate(
    pdf_path="document.pdf",
    output_path="podcast.mp3",
    complexity="intermediate",  # Options: "simple", "intermediate", "advanced"
    voice_id="Joanna"  # Optional: override default voice
)

# Access results
print(f"Generated podcast: {result['audio']['path']}")
print(f"Audio size: {result['audio']['size']} bytes")
print(f"Script length: {len(result['script'])} characters")
```

## Available Providers

### LLM Providers
- `"gemini"`: Google's Gemini LLM
  - Requires: GENAI_API_KEY
  - Configuration options:
    - model_name: Model version to use
    - temperature: Output randomness (0-1)
    - max_output_tokens: Maximum output length
    - top_p: Nucleus sampling parameter
    - streaming: Enable/disable streaming mode
    - prompt_builder: Custom prompt builder instance

### TTS Providers
- `"aws"`: Amazon Polly
  - Requires: AWS credentials
  - Configuration options:
    - voice_id: Voice to use (e.g., "Joanna", "Matthew")
    - region_name: AWS region
    - engine: "standard" or "neural"
    
- `"google"`: Google Text-to-Speech
  - No API key required
  - Configuration options:
    - language: Language code (e.g., "en", "es")
    - tld: Top-level domain for accent (e.g., "com", "co.uk")
    - slow: Speech speed

## PDF Processing Features

The library offers advanced PDF processing capabilities:

### Basic Features
- Metadata extraction (title, author, subject, keywords)
- Image caption extraction from documents
- Efficient processing of large documents
- Support for complex PDF layouts

### Text Processing
- Smart text chunking with customizable size
- Paragraph-aware text splitting
- Sentence boundary preservation

### Semantic Search & Retrieval
- Vector-based semantic search using FAISS
- Embedding generation with Sentence Transformers
- Retrieval of relevant text chunks based on queries

## Configuration

### Environment Variables

You can set these environment variables instead of passing them directly:

```
GENAI_API_KEY=your_google_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=your_aws_region
```

### Advanced Configuration Examples

#### High-Quality Production Setup
```python
generator = PodcastGenerator(
    rag_system=processor,
    llm_type="gemini",
    tts_type="aws",
    llm_config={
        "model_name": "gemini-1.5-flash",
        "temperature": 0.2,
        "top_p": 0.9,
        "max_output_tokens": 8192,
        "streaming": True
    },
    tts_config={
        "voice_id": "Joanna",
        "engine": "neural",
        "region_name": "us-west-2"
    }
)
```

#### Fast Development Setup
```python
generator = PodcastGenerator(
    rag_system=processor,
    llm_type="gemini",
    tts_type="google",  # Faster, no API key needed
    llm_config={
        "temperature": 0.3,
        "max_output_tokens": 4096
    },
    tts_config={
        "language": "en",
        "tld": "com"
    }
)
```

## Custom Prompt Builders

You can customize how content is processed by creating custom prompt builders:

```python
from pdf2podcast.core.base import BasePromptBuilder

class TechnicalPromptBuilder(BasePromptBuilder):
    """Specialized for technical documentation."""
    
    def build_prompt(self, text: str, **kwargs) -> str:
        return (
            "Create a technical podcast script following these guidelines:\n"
            "1. Start with a high-level overview\n"
            "2. Break down complex concepts step by step\n"
            "3. Include practical examples\n\n"
            f"Content: {text}\n"
            f"Complexity: {kwargs.get('complexity', 'intermediate')}"
        )

# Use custom prompt builder
generator = PodcastGenerator(
    rag_system=processor,
    llm_type="gemini",
    tts_type="aws",
    llm_config={
        "prompt_builder": TechnicalPromptBuilder(),
        "temperature": 0.2
    }
)
```

## Common Use Cases

### Academic Paper Processing
```python
generator = PodcastGenerator(
    rag_system=SimplePDFProcessor(
        max_chars_per_chunk=8000,
        extract_images=True,
        metadata=True
    ),
    llm_type="gemini",
    tts_type="aws",
    llm_config={
        "temperature": 0.2,
        "max_output_tokens": 8192
    },
    tts_config={
        "voice_id": "Joanna",
        "engine": "neural"
    }
)

result = generator.generate(
    pdf_path="paper.pdf",
    output_path="paper_podcast.mp3",
    complexity="advanced",
    query="Focus on methodology and key findings"
)
```

### Business Report Summary
```python
generator = PodcastGenerator(
    rag_system=SimplePDFProcessor(
        max_chars_per_chunk=4000
    ),
    llm_type="gemini",
    tts_type="aws",
    llm_config={
        "temperature": 0.3,
        "max_output_tokens": 4096
    }
)

result = generator.generate(
    pdf_path="report.pdf",
    output_path="summary.mp3",
    complexity="intermediate",
    query="Summarize key business metrics and trends"
)
```

### Educational Content
```python
generator = PodcastGenerator(
    rag_system=SimplePDFProcessor(extract_images=True),
    llm_type="gemini",
    tts_type="google",
    llm_config={
        "temperature": 0.4,
        "max_output_tokens": 6144
    },
    tts_config={
        "language": "en",
        "tld": "com",
        "slow": True  # Better for learning
    }
)

result = generator.generate(
    pdf_path="lesson.pdf",
    output_path="tutorial.mp3",
    complexity="simple"
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
