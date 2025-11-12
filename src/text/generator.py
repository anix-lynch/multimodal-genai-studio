"""
Text Generation Module
Supports multiple LLM providers: Google Gemini, OpenAI, Anthropic
"""

import logging
from typing import Optional, Dict, Any
import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic

from config import APIKeys, ModelConfig, GenerationConfig

logger = logging.getLogger(__name__)


class TextGenerator:
    """Multi-provider text generation with unified interface"""
    
    def __init__(self):
        """Initialize text generator with available providers"""
        self.providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Setup API clients for available providers"""
        
        # Google Gemini
        if APIKeys.GOOGLE_API_KEY:
            try:
                genai.configure(api_key=APIKeys.GOOGLE_API_KEY)
                self.providers["google"] = genai
                logger.info("✅ Google Gemini initialized")
            except Exception as e:
                logger.error(f"❌ Google Gemini initialization failed: {e}")
        
        # OpenAI
        if APIKeys.OPENAI_API_KEY:
            try:
                self.providers["openai"] = OpenAI(api_key=APIKeys.OPENAI_API_KEY)
                logger.info("✅ OpenAI initialized")
            except Exception as e:
                logger.error(f"❌ OpenAI initialization failed: {e}")
        
        # Anthropic
        if APIKeys.ANTHROPIC_API_KEY:
            try:
                self.providers["anthropic"] = Anthropic(api_key=APIKeys.ANTHROPIC_API_KEY)
                logger.info("✅ Anthropic initialized")
            except Exception as e:
                logger.error(f"❌ Anthropic initialization failed: {e}")
    
    def get_available_models(self) -> list[str]:
        """Get list of available text models"""
        available = []
        for model_name, config in ModelConfig.TEXT_MODELS.items():
            provider = config["provider"]
            if provider in self.providers:
                available.append(model_name)
        return available
    
    def generate(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = None,
        temperature: float = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using specified model
        
        Args:
            prompt: User prompt
            model: Model name (e.g., "gemini-pro", "gpt-4o")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            system_prompt: Optional system instructions
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dict with 'text', 'model', 'tokens' keys
        """
        
        # Use defaults if not specified
        model = model or ModelConfig.DEFAULT_TEXT_MODEL
        max_tokens = max_tokens or GenerationConfig.MAX_TOKENS
        temperature = temperature if temperature is not None else GenerationConfig.TEMPERATURE
        
        # Check if model is available
        if model not in self.get_available_models():
            raise ValueError(f"Model '{model}' not available. Available models: {self.get_available_models()}")
        
        # Get provider
        model_config = ModelConfig.TEXT_MODELS[model]
        provider = model_config["provider"]
        
        logger.info(f"Generating text with {model} (provider: {provider})")
        
        try:
            if provider == "google":
                return self._generate_google(prompt, model, max_tokens, temperature, system_prompt)
            elif provider == "openai":
                return self._generate_openai(prompt, model, max_tokens, temperature, system_prompt)
            elif provider == "anthropic":
                return self._generate_anthropic(prompt, model, max_tokens, temperature, system_prompt)
            else:
                raise ValueError(f"Unknown provider: {provider}")
                
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    def _generate_google(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Generate using Google Gemini"""
        
        genai_model = self.providers["google"].GenerativeModel(
            model_name=model,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        
        # Combine system prompt with user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = genai_model.generate_content(full_prompt)
        
        return {
            "text": response.text,
            "model": model,
            "tokens": len(response.text.split()),  # Approximate
            "provider": "google"
        }
    
    def _generate_openai(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Generate using OpenAI"""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.providers["openai"].chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        return {
            "text": response.choices[0].message.content,
            "model": model,
            "tokens": response.usage.total_tokens,
            "provider": "openai"
        }
    
    def _generate_anthropic(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Generate using Anthropic Claude"""
        
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = self.providers["anthropic"].messages.create(**kwargs)
        
        return {
            "text": response.content[0].text,
            "model": model,
            "tokens": response.usage.input_tokens + response.usage.output_tokens,
            "provider": "anthropic"
        }
    
    def stream_generate(
        self,
        prompt: str,
        model: str = None,
        **kwargs
    ):
        """
        Stream text generation (for real-time output)
        Yields text chunks as they're generated
        """
        
        model = model or ModelConfig.DEFAULT_TEXT_MODEL
        model_config = ModelConfig.TEXT_MODELS[model]
        provider = model_config["provider"]
        
        # Streaming support varies by provider
        if provider == "google":
            genai_model = self.providers["google"].GenerativeModel(model_name=model)
            response = genai_model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        
        elif provider == "openai":
            stream = self.providers["openai"].chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        else:
            # Fall back to non-streaming
            result = self.generate(prompt, model, **kwargs)
            yield result["text"]


# Example usage
if __name__ == "__main__":
    generator = TextGenerator()
    
    print("📝 Available models:", generator.get_available_models())
    
    if generator.get_available_models():
        result = generator.generate(
            prompt="Write a haiku about AI",
            temperature=0.7
        )
        print(f"\n✨ Generated text ({result['model']}):")
        print(result["text"])
        print(f"\n📊 Tokens used: {result['tokens']}")

