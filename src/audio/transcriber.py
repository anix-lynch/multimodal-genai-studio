"""
Audio Transcription Module
Supports OpenAI Whisper (API and local)
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from openai import OpenAI
from config import APIKeys, ModelConfig, OUTPUT_DIR

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """Audio transcription with multiple providers"""
    
    def __init__(self):
        """Initialize audio transcriber"""
        self.providers = {}
        self._initialize_providers()
        self.output_dir = OUTPUT_DIR / "transcriptions"
        self.output_dir.mkdir(exist_ok=True)
    
    def _initialize_providers(self):
        """Setup API clients for available providers"""
        
        # OpenAI Whisper API
        if APIKeys.OPENAI_API_KEY:
            try:
                self.providers["openai"] = OpenAI(api_key=APIKeys.OPENAI_API_KEY)
                logger.info("✅ OpenAI Whisper API initialized")
            except Exception as e:
                logger.error(f"❌ OpenAI initialization failed: {e}")
        
        # Local Whisper
        try:
            import whisper
            self.providers["local"] = whisper
            logger.info("✅ Local Whisper initialized")
        except ImportError:
            logger.warning("Local Whisper not available (install openai-whisper)")
    
    def get_available_models(self) -> list[str]:
        """Get list of available transcription models"""
        available = []
        
        if "openai" in self.providers:
            available.append("whisper-1")
        
        if "local" in self.providers:
            available.extend(["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large"])
        
        return available
    
    def transcribe(
        self,
        audio_path: str,
        model: str = "whisper-1",
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file (mp3, mp4, wav, etc.)
            model: Model name (e.g., "whisper-1", "whisper-base")
            language: Optional language code (e.g., "en", "ja")
            task: "transcribe" or "translate" (translate to English)
            
        Returns:
            Dict with 'text', 'model', 'language' keys
        """
        
        if model not in self.get_available_models():
            raise ValueError(f"Model '{model}' not available. Available models: {self.get_available_models()}")
        
        logger.info(f"Transcribing audio with {model}")
        
        try:
            if model == "whisper-1":
                return self._transcribe_openai(audio_path, language, task)
            else:
                return self._transcribe_local(audio_path, model, language, task)
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def _transcribe_openai(
        self,
        audio_path: str,
        language: Optional[str],
        task: str
    ) -> Dict[str, Any]:
        """Transcribe using OpenAI Whisper API"""
        
        with open(audio_path, "rb") as audio_file:
            if task == "translate":
                response = self.providers["openai"].audio.translations.create(
                    model="whisper-1",
                    file=audio_file,
                )
            else:
                kwargs = {
                    "model": "whisper-1",
                    "file": audio_file,
                }
                if language:
                    kwargs["language"] = language
                
                response = self.providers["openai"].audio.transcriptions.create(**kwargs)
        
        # Save transcription
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"transcription_{timestamp}.txt"
        output_file.write_text(response.text)
        
        logger.info(f"✅ Transcription saved: {output_file}")
        
        return {
            "text": response.text,
            "model": "whisper-1",
            "language": language or "auto",
            "output_file": str(output_file),
            "provider": "openai"
        }
    
    def _transcribe_local(
        self,
        audio_path: str,
        model: str,
        language: Optional[str],
        task: str
    ) -> Dict[str, Any]:
        """Transcribe using local Whisper model"""
        
        # Extract model size (e.g., "whisper-base" -> "base")
        model_size = model.replace("whisper-", "")
        
        # Load model
        whisper_model = self.providers["local"].load_model(model_size)
        
        # Transcribe
        kwargs = {}
        if language:
            kwargs["language"] = language
        if task == "translate":
            kwargs["task"] = "translate"
        
        result = whisper_model.transcribe(audio_path, **kwargs)
        
        # Save transcription
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"transcription_{timestamp}.txt"
        output_file.write_text(result["text"])
        
        logger.info(f"✅ Transcription saved: {output_file}")
        
        return {
            "text": result["text"],
            "model": model,
            "language": result.get("language", language or "auto"),
            "output_file": str(output_file),
            "provider": "local"
        }
    
    def transcribe_with_timestamps(
        self,
        audio_path: str,
        model: str = "whisper-base"
    ) -> Dict[str, Any]:
        """
        Transcribe with word-level timestamps (local only)
        
        Args:
            audio_path: Path to audio file
            model: Local Whisper model
            
        Returns:
            Dict with 'segments' containing timestamped text
        """
        
        if "local" not in self.providers:
            raise ValueError("Timestamp transcription requires local Whisper")
        
        model_size = model.replace("whisper-", "")
        whisper_model = self.providers["local"].load_model(model_size)
        
        result = whisper_model.transcribe(audio_path, word_timestamps=True)
        
        # Format segments
        segments = []
        for segment in result["segments"]:
            segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })
        
        # Save with timestamps
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"transcription_timestamps_{timestamp}.txt"
        
        with open(output_file, "w") as f:
            for seg in segments:
                f.write(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}\n")
        
        logger.info(f"✅ Timestamped transcription saved: {output_file}")
        
        return {
            "text": result["text"],
            "segments": segments,
            "model": model,
            "output_file": str(output_file),
            "provider": "local"
        }


# Example usage
if __name__ == "__main__":
    transcriber = AudioTranscriber()
    
    print("🎤 Available models:", transcriber.get_available_models())

