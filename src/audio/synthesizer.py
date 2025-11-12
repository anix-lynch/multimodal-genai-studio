"""
Text-to-Speech Module
Supports OpenAI TTS and Google TTS (gTTS)
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from openai import OpenAI
from gtts import gTTS
from config import APIKeys, OUTPUT_DIR

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Text-to-speech synthesis with multiple providers"""
    
    def __init__(self):
        """Initialize TTS engine"""
        self.providers = {}
        self._initialize_providers()
        self.output_dir = OUTPUT_DIR / "audio"
        self.output_dir.mkdir(exist_ok=True)
    
    def _initialize_providers(self):
        """Setup API clients for available providers"""
        
        # OpenAI TTS
        if APIKeys.OPENAI_API_KEY:
            try:
                self.providers["openai"] = OpenAI(api_key=APIKeys.OPENAI_API_KEY)
                logger.info("✅ OpenAI TTS initialized")
            except Exception as e:
                logger.error(f"❌ OpenAI initialization failed: {e}")
        
        # Google TTS (always available)
        self.providers["gtts"] = True
        logger.info("✅ Google TTS (gTTS) initialized")
    
    def get_available_models(self) -> list[str]:
        """Get list of available TTS models"""
        available = []
        
        if "openai" in self.providers:
            available.extend(["tts-1", "tts-1-hd"])
        
        if "gtts" in self.providers:
            available.append("gtts")
        
        return available
    
    def get_available_voices(self, model: str = "tts-1") -> list[str]:
        """Get available voices for a model"""
        
        if model in ["tts-1", "tts-1-hd"]:
            return ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        elif model == "gtts":
            # gTTS supports many languages, but no voice selection
            return ["default"]
        
        return []
    
    def synthesize(
        self,
        text: str,
        model: str = "gtts",
        voice: str = "default",
        language: str = "en",
        speed: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convert text to speech
        
        Args:
            text: Text to synthesize
            model: Model name (e.g., "tts-1", "gtts")
            voice: Voice name (OpenAI: alloy, echo, fable, onyx, nova, shimmer)
            language: Language code (e.g., "en", "ja")
            speed: Speech speed (0.25 to 4.0 for OpenAI)
            
        Returns:
            Dict with 'audio_path', 'model', 'duration' keys
        """
        
        if model not in self.get_available_models():
            raise ValueError(f"Model '{model}' not available. Available models: {self.get_available_models()}")
        
        logger.info(f"Synthesizing speech with {model}")
        
        try:
            if model in ["tts-1", "tts-1-hd"]:
                return self._synthesize_openai(text, model, voice, speed)
            elif model == "gtts":
                return self._synthesize_gtts(text, language)
            else:
                raise ValueError(f"Unknown model: {model}")
                
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            raise
    
    def _synthesize_openai(
        self,
        text: str,
        model: str,
        voice: str,
        speed: float
    ) -> Dict[str, Any]:
        """Synthesize using OpenAI TTS"""
        
        # Validate voice
        available_voices = self.get_available_voices(model)
        if voice not in available_voices:
            logger.warning(f"Voice '{voice}' not available for {model}, using 'alloy'")
            voice = "alloy"
        
        # Clamp speed
        speed = max(0.25, min(4.0, speed))
        
        response = self.providers["openai"].audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            speed=speed,
        )
        
        # Save audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_{model}_{voice}_{timestamp}.mp3"
        filepath = self.output_dir / filename
        
        response.stream_to_file(str(filepath))
        
        logger.info(f"✅ Audio saved: {filepath}")
        
        return {
            "audio_path": str(filepath),
            "model": model,
            "voice": voice,
            "speed": speed,
            "provider": "openai"
        }
    
    def _synthesize_gtts(
        self,
        text: str,
        language: str
    ) -> Dict[str, Any]:
        """Synthesize using Google TTS (gTTS)"""
        
        tts = gTTS(text=text, lang=language, slow=False)
        
        # Save audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_gtts_{language}_{timestamp}.mp3"
        filepath = self.output_dir / filename
        
        tts.save(str(filepath))
        
        logger.info(f"✅ Audio saved: {filepath}")
        
        return {
            "audio_path": str(filepath),
            "model": "gtts",
            "language": language,
            "provider": "gtts"
        }
    
    def synthesize_long_form(
        self,
        text: str,
        model: str = "tts-1",
        voice: str = "alloy",
        chunk_size: int = 4096
    ) -> Dict[str, Any]:
        """
        Synthesize long text by chunking (for texts > 4096 chars)
        
        Args:
            text: Long text to synthesize
            model: TTS model
            voice: Voice name
            chunk_size: Maximum characters per chunk
            
        Returns:
            Dict with audio path of combined audio
        """
        
        if len(text) <= chunk_size:
            return self.synthesize(text, model, voice)
        
        # Split into chunks (at sentence boundaries)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"Split into {len(chunks)} chunks")
        
        # Synthesize each chunk
        audio_files = []
        for i, chunk in enumerate(chunks):
            result = self.synthesize(chunk, model, voice)
            audio_files.append(result["audio_path"])
            logger.info(f"Chunk {i+1}/{len(chunks)} complete")
        
        # Combine audio files
        from pydub import AudioSegment
        
        combined = AudioSegment.empty()
        for audio_file in audio_files:
            audio = AudioSegment.from_mp3(audio_file)
            combined += audio
        
        # Save combined audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_longform_{timestamp}.mp3"
        filepath = self.output_dir / filename
        
        combined.export(str(filepath), format="mp3")
        
        logger.info(f"✅ Combined audio saved: {filepath}")
        
        return {
            "audio_path": str(filepath),
            "model": model,
            "voice": voice,
            "chunks": len(chunks),
            "provider": "openai"
        }


# Example usage
if __name__ == "__main__":
    tts = TextToSpeech()
    
    print("🔊 Available models:", tts.get_available_models())
    print("🎙️ Available voices (tts-1):", tts.get_available_voices("tts-1"))
    
    if tts.get_available_models():
        result = tts.synthesize(
            text="Hello! This is a test of the text-to-speech system.",
            model="gtts",
            language="en"
        )
        print(f"\n✨ Audio generated:")
        print(f"  📁 {result['audio_path']}")

