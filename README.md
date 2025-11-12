# 🎨 Multimodal GenAI Studio

A professional-grade multimodal AI application that generates and processes content across text, image, and audio modalities. Built to showcase expertise in multimodal generative AI and modern AI application development.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

### Text Generation
- **Multiple LLM Support**: Google Gemini, OpenAI GPT-4o, Anthropic Claude
- **Flexible Parameters**: Control temperature, max tokens, system prompts
- **Streaming Support**: Real-time text generation
- **FREE Option**: Gemini provides 60 requests/minute free

### Image Generation
- **DALL-E Integration**: DALL-E 2 and 3 with quality/style controls
- **Stable Diffusion**: Via HuggingFace API
- **Image Editing**: Edit existing images with prompts
- **Variations**: Generate variations of images
- **Multiple Formats**: Support for various sizes and aspect ratios

### Audio Processing
- **Transcription**: OpenAI Whisper API and local models
- **Multi-language**: Support for 50+ languages
- **Translation**: Automatic translation to English
- **Text-to-Speech**: OpenAI TTS (6 voices) and Google TTS
- **Voice Control**: Multiple voice options and speed control

### Multimodal Pipelines
- **Story to Multimedia**: Generate scene images + audio narration
- **Audio to Blog**: Transcribe audio → formatted blog post + featured image
- **Creative Chains**: Iterative image→text→image workflows
- **Text Roundtrip**: Test TTS→transcription accuracy

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- At least ONE API key (Google, OpenAI, or Anthropic)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/multimodal-genai-studio
cd multimodal-genai-studio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

### Minimal Setup (FREE)

For a completely FREE setup, you only need:

```bash
# .env
GOOGLE_API_KEY=your_gemini_api_key
```

Get a FREE Gemini API key: https://makersuite.google.com/app/apikey

This enables:
- ✅ Text generation (Gemini)
- ✅ Text-to-speech (gTTS built-in)
- ✅ Basic multimodal pipelines

### Run Application

```bash
python app.py
```

Open browser to: http://localhost:7861

## 📚 Documentation

### Configuration

The app supports multiple API providers. Configure in `.env`:

```bash
# === Required (at least one) ===
GOOGLE_API_KEY=your_key          # Gemini (FREE: 60 req/min)
OPENAI_API_KEY=your_key          # GPT, DALL-E, Whisper
ANTHROPIC_API_KEY=your_key       # Claude

# === Optional ===
HF_TOKEN=your_token              # Stable Diffusion
STABILITY_API_KEY=your_key       # Stability AI

# === Server Settings ===
HOST=0.0.0.0
PORT=7861
```

### Usage Examples

#### Text Generation
```python
from src.text.generator import TextGenerator

gen = TextGenerator()
result = gen.generate(
    prompt="Write a haiku about AI",
    model="gemini-1.5-flash",
    temperature=0.7
)
print(result['text'])
```

#### Image Generation
```python
from src.image.generator import ImageGenerator

gen = ImageGenerator()
result = gen.generate(
    prompt="A serene mountain landscape at sunset",
    model="dall-e-3",
    size="1024x1024"
)
print(f"Image saved to: {result['images'][0]}")
```

#### Audio Transcription
```python
from src.audio.transcriber import AudioTranscriber

trans = AudioTranscriber()
result = trans.transcribe(
    audio_path="audio.mp3",
    model="whisper-1"
)
print(result['text'])
```

#### Text-to-Speech
```python
from src.audio.synthesizer import TextToSpeech

tts = TextToSpeech()
result = tts.synthesize(
    text="Hello, this is a test",
    model="gtts",  # FREE option
    language="en"
)
print(f"Audio saved to: {result['audio_path']}")
```

#### Multimodal Pipeline
```python
from src.multimodal.pipeline import MultimodalPipeline

pipeline = MultimodalPipeline()
result = pipeline.story_to_multimedia(
    story="Once upon a time...",
    generate_images=True,
    generate_audio=True
)
print(f"Generated {len(result['images'])} images")
print(f"Audio: {result['audio_path']}")
```

## 🏗️ Architecture

```
multimodal-genai-studio/
├── app.py                          # Main Gradio application
├── config.py                       # Configuration management
├── requirements.txt                # Python dependencies
│
├── src/                            # Source code
│   ├── text/                       # Text generation
│   │   └── generator.py            # Multi-provider LLM
│   ├── image/                      # Image generation
│   │   └── generator.py            # DALL-E + Stable Diffusion
│   ├── audio/                      # Audio processing
│   │   ├── transcriber.py          # Speech-to-text
│   │   └── synthesizer.py          # Text-to-speech
│   └── multimodal/                 # Multimodal workflows
│       └── pipeline.py             # Combined pipelines
│
├── outputs/                        # Generated content
│   ├── images/                     # Generated images
│   ├── audio/                      # Generated audio
│   ├── transcriptions/             # Transcription outputs
│   └── multimodal/                 # Pipeline outputs
│
└── deployment/                     # Deployment configs
    ├── Dockerfile                  # Docker configuration
    ├── docker-compose.yml          # Docker Compose
    └── README_HF_SPACES.md         # HuggingFace deployment
```

## 🎯 Key Technologies

- **LLMs**: Google Gemini, OpenAI GPT-4o, Anthropic Claude
- **Image Generation**: DALL-E 2/3, Stable Diffusion XL
- **Audio**: OpenAI Whisper, OpenAI TTS, gTTS
- **Framework**: Gradio (Modern UI)
- **Backend**: Python 3.11+
- **Async**: Concurrent processing support

## 🚢 Deployment

### Docker

```bash
docker-compose -f deployment/docker-compose.yml up
```

### HuggingFace Spaces

See [deployment/README_HF_SPACES.md](deployment/README_HF_SPACES.md) for detailed instructions.

Quick deploy:
1. Create Space on HuggingFace
2. Upload project files
3. Set API keys in secrets
4. Deploy (FREE on CPU Basic)

### Local Production

```bash
# Install production dependencies
pip install -r requirements.txt

# Run with gunicorn (recommended)
gunicorn app:app --bind 0.0.0.0:7861 --workers 4
```

## 💰 Cost Analysis

### FREE Tier (Recommended for Personal Use)
- **Google Gemini**: 60 requests/min FREE
- **gTTS**: Unlimited FREE
- **HuggingFace**: Inference API FREE tier
- **Hosting**: HuggingFace Spaces CPU Basic FREE

**Total: $0/month** ✅

### Paid Tier (Production)
- **OpenAI GPT-4o**: ~$10-50/month (typical usage)
- **DALL-E 3**: $0.04 per image
- **Whisper**: $0.006 per minute
- **TTS**: $15 per 1M characters
- **Hosting**: HuggingFace CPU Upgrade ~$22/month

**Total: $30-100/month** (varies by usage)

## 🎓 Skills Demonstrated

This project showcases expertise in:

### AI/ML
- ✅ Multimodal AI application development
- ✅ LLM integration and prompt engineering
- ✅ Image generation and manipulation
- ✅ Audio processing and synthesis
- ✅ Pipeline orchestration

### Software Engineering
- ✅ Clean architecture and modular design
- ✅ Error handling and logging
- ✅ Configuration management
- ✅ API integration best practices
- ✅ Production-ready code

### DevOps
- ✅ Docker containerization
- ✅ Environment management
- ✅ Deployment automation
- ✅ Multi-platform deployment

### Full Stack
- ✅ Modern UI with Gradio
- ✅ Backend API design
- ✅ File handling and storage
- ✅ Async processing

## 📊 Certifications

This project demonstrates skills from:
- **Build Multimodal Generative AI Applications** (IBM)
- Python for Data Science, AI & Development (IBM)
- Fundamentals of Building AI Agents (IBM)

## 🔧 Development

### Run Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black src/ app.py config.py

# Type checking
mypy src/

# Linting
pylint src/
```

### Add New Models

To add a new LLM provider:
1. Update `ModelConfig.TEXT_MODELS` in `config.py`
2. Add provider initialization in `TextGenerator.__init__`
3. Implement `_generate_{provider}` method

## 🤝 Contributing

Contributions welcome! Please:
1. Fork repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Google Gemini for FREE LLM access
- OpenAI for powerful multimodal APIs
- Anthropic for Claude
- HuggingFace for Stable Diffusion
- Gradio for amazing UI framework

## 📞 Support

- **Documentation**: See `/deployment` folder
- **Issues**: [GitHub Issues](https://github.com/yourusername/multimodal-genai-studio/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/multimodal-genai-studio/discussions)

## 🚀 Roadmap

- [ ] Video generation support
- [ ] Real-time streaming UI
- [ ] Multi-user support
- [ ] API endpoints
- [ ] More pipeline templates
- [ ] Fine-tuning support

---

**Built with ❤️ to showcase multimodal AI expertise**

**Demo**: [Live Demo URL]  
**Portfolio**: [Your Portfolio]  
**LinkedIn**: [Your LinkedIn]

