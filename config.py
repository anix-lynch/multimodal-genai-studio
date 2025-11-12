"""
Multimodal GenAI Studio - Configuration
Centralizes all configuration and API key management
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Paths ===
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"
CACHE_DIR = BASE_DIR / "cache"

# Create directories
for dir_path in [OUTPUT_DIR, UPLOAD_DIR, CACHE_DIR]:
    dir_path.mkdir(exist_ok=True)

# === API Keys ===
class APIKeys:
    """Centralized API key management"""
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "")
    
    @classmethod
    def validate(cls):
        """Check which APIs are available"""
        available = {
            "google": bool(cls.GOOGLE_API_KEY),
            "openai": bool(cls.OPENAI_API_KEY),
            "anthropic": bool(cls.ANTHROPIC_API_KEY),
            "huggingface": bool(cls.HF_TOKEN),
            "stability": bool(cls.STABILITY_API_KEY),
        }
        return available

# === Model Configuration ===
class ModelConfig:
    """Model settings and defaults"""
    
    # Text Models
    TEXT_MODELS = {
        "gemini-pro": {"provider": "google", "max_tokens": 8192},
        "gemini-1.5-flash": {"provider": "google", "max_tokens": 8192},
        "gpt-4o": {"provider": "openai", "max_tokens": 4096},
        "gpt-4o-mini": {"provider": "openai", "max_tokens": 4096},
        "claude-3-5-sonnet-20241022": {"provider": "anthropic", "max_tokens": 8192},
        "claude-3-5-haiku-20241022": {"provider": "anthropic", "max_tokens": 8192},
    }
    
    # Image Models
    IMAGE_MODELS = {
        "dall-e-3": {"provider": "openai", "sizes": ["1024x1024", "1792x1024", "1024x1792"]},
        "dall-e-2": {"provider": "openai", "sizes": ["256x256", "512x512", "1024x1024"]},
        "stable-diffusion-xl": {"provider": "huggingface", "sizes": ["1024x1024"]},
    }
    
    # Audio Models
    AUDIO_MODELS = {
        "whisper-1": {"provider": "openai", "task": "transcription"},
        "whisper-base": {"provider": "local", "task": "transcription"},
        "tts-1": {"provider": "openai", "task": "synthesis"},
        "gtts": {"provider": "local", "task": "synthesis"},
    }
    
    # Defaults
    DEFAULT_TEXT_MODEL = os.getenv("DEFAULT_TEXT_MODEL", "gemini-1.5-flash")
    DEFAULT_IMAGE_MODEL = os.getenv("DEFAULT_IMAGE_MODEL", "dall-e-3")
    DEFAULT_AUDIO_MODEL = os.getenv("DEFAULT_AUDIO_MODEL", "whisper-1")

# === Generation Settings ===
class GenerationConfig:
    """Default generation parameters"""
    
    # Text Generation
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
    TEMPERATURE = 0.7
    TOP_P = 0.9
    
    # Image Generation
    MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1024"))
    IMAGE_QUALITY = "standard"  # standard or hd
    IMAGE_STYLE = "vivid"  # vivid or natural
    
    # Audio Processing
    MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", "300"))  # seconds
    AUDIO_FORMAT = "mp3"
    SAMPLE_RATE = 22050

# === Server Configuration ===
class ServerConfig:
    """Server settings"""
    
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "7861"))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Gradio settings
    SHARE = False
    SERVER_NAME = HOST
    SERVER_PORT = PORT

# === Logging ===
class LogConfig:
    """Logging configuration"""
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = BASE_DIR / "logs" / "app.log"
    
    @classmethod
    def setup(cls):
        """Setup logging"""
        import logging
        
        cls.LOG_FILE.parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(cls.LOG_FILE),
                logging.StreamHandler()
            ]
        )

# Initialize logging
LogConfig.setup()

# === Validation ===
def get_available_features():
    """Check which features are available based on API keys"""
    api_status = APIKeys.validate()
    
    features = {
        "text_generation": api_status["google"] or api_status["openai"] or api_status["anthropic"],
        "image_generation": api_status["openai"] or api_status["huggingface"] or api_status["stability"],
        "audio_transcription": api_status["openai"] or True,  # Local whisper always available
        "audio_synthesis": api_status["openai"] or True,  # gTTS always available
        "multimodal": api_status["google"] or api_status["openai"],
    }
    
    return features, api_status

if __name__ == "__main__":
    # Print configuration summary
    print("🚀 Multimodal GenAI Studio Configuration")
    print("=" * 50)
    
    features, api_status = get_available_features()
    
    print("\n📊 API Status:")
    for api, status in api_status.items():
        print(f"  {api.capitalize()}: {'✅' if status else '❌'}")
    
    print("\n🎯 Available Features:")
    for feature, available in features.items():
        print(f"  {feature.replace('_', ' ').title()}: {'✅' if available else '❌'}")
    
    print("\n📁 Directories:")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Upload: {UPLOAD_DIR}")
    print(f"  Cache: {CACHE_DIR}")
    
    print("\n🔧 Server:")
    print(f"  Host: {ServerConfig.HOST}")
    print(f"  Port: {ServerConfig.PORT}")
    
    print("\n" + "=" * 50)

