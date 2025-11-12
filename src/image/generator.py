"""
Image Generation Module
Supports DALL-E (OpenAI) and Stable Diffusion (HuggingFace/Local)
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime

from openai import OpenAI
from config import APIKeys, ModelConfig, GenerationConfig, OUTPUT_DIR

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Multi-provider image generation"""
    
    def __init__(self):
        """Initialize image generator with available providers"""
        self.providers = {}
        self._initialize_providers()
        self.output_dir = OUTPUT_DIR / "images"
        self.output_dir.mkdir(exist_ok=True)
    
    def _initialize_providers(self):
        """Setup API clients for available providers"""
        
        # OpenAI DALL-E
        if APIKeys.OPENAI_API_KEY:
            try:
                self.providers["openai"] = OpenAI(api_key=APIKeys.OPENAI_API_KEY)
                logger.info("✅ OpenAI DALL-E initialized")
            except Exception as e:
                logger.error(f"❌ OpenAI initialization failed: {e}")
        
        # HuggingFace (for Stable Diffusion)
        if APIKeys.HF_TOKEN:
            try:
                self.providers["huggingface"] = APIKeys.HF_TOKEN
                logger.info("✅ HuggingFace initialized")
            except Exception as e:
                logger.error(f"❌ HuggingFace initialization failed: {e}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available image models"""
        available = []
        for model_name, config in ModelConfig.IMAGE_MODELS.items():
            provider = config["provider"]
            if provider in self.providers:
                available.append(model_name)
        return available
    
    def generate(
        self,
        prompt: str,
        model: str = None,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
        n: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate image from text prompt
        
        Args:
            prompt: Text description of desired image
            model: Model name (e.g., "dall-e-3", "stable-diffusion-xl")
            size: Image size (e.g., "1024x1024", "1792x1024")
            quality: "standard" or "hd" (DALL-E only)
            style: "vivid" or "natural" (DALL-E only)
            n: Number of images to generate
            
        Returns:
            Dict with 'images' (list of paths), 'model', 'prompt' keys
        """
        
        # Use defaults if not specified
        model = model or ModelConfig.DEFAULT_IMAGE_MODEL
        
        # Check if model is available
        if model not in self.get_available_models():
            raise ValueError(f"Model '{model}' not available. Available models: {self.get_available_models()}")
        
        # Get provider
        model_config = ModelConfig.IMAGE_MODELS[model]
        provider = model_config["provider"]
        
        logger.info(f"Generating image with {model} (provider: {provider})")
        
        try:
            if provider == "openai":
                return self._generate_openai(prompt, model, size, quality, style, n)
            elif provider == "huggingface":
                return self._generate_huggingface(prompt, model, size, n)
            else:
                raise ValueError(f"Unknown provider: {provider}")
                
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise
    
    def _generate_openai(
        self,
        prompt: str,
        model: str,
        size: str,
        quality: str,
        style: str,
        n: int
    ) -> Dict[str, Any]:
        """Generate using OpenAI DALL-E"""
        
        # Validate size for model
        valid_sizes = ModelConfig.IMAGE_MODELS[model]["sizes"]
        if size not in valid_sizes:
            logger.warning(f"Size {size} not valid for {model}, using {valid_sizes[0]}")
            size = valid_sizes[0]
        
        # DALL-E 3 only supports n=1
        if model == "dall-e-3" and n > 1:
            logger.warning("DALL-E 3 only supports generating 1 image at a time")
            n = 1
        
        response = self.providers["openai"].images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality if model == "dall-e-3" else "standard",
            style=style if model == "dall-e-3" else None,
            n=n,
        )
        
        # Download and save images
        image_paths = []
        for i, image_data in enumerate(response.data):
            # Download image
            image_response = requests.get(image_data.url)
            img = Image.open(BytesIO(image_response.content))
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model}_{timestamp}_{i}.png"
            filepath = self.output_dir / filename
            img.save(filepath)
            image_paths.append(str(filepath))
            
            logger.info(f"✅ Image saved: {filepath}")
        
        return {
            "images": image_paths,
            "model": model,
            "prompt": prompt,
            "revised_prompt": response.data[0].revised_prompt if hasattr(response.data[0], 'revised_prompt') else None,
            "size": size,
            "provider": "openai"
        }
    
    def _generate_huggingface(
        self,
        prompt: str,
        model: str,
        size: str,
        n: int
    ) -> Dict[str, Any]:
        """Generate using HuggingFace Inference API"""
        
        API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {self.providers['huggingface']}"}
        
        image_paths = []
        
        for i in range(n):
            response = requests.post(
                API_URL,
                headers=headers,
                json={"inputs": prompt}
            )
            
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                
                # Resize if needed
                width, height = map(int, size.split('x'))
                img = img.resize((width, height))
                
                # Save with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"stable-diffusion_{timestamp}_{i}.png"
                filepath = self.output_dir / filename
                img.save(filepath)
                image_paths.append(str(filepath))
                
                logger.info(f"✅ Image saved: {filepath}")
            else:
                logger.error(f"HuggingFace API error: {response.status_code}")
                raise Exception(f"Image generation failed: {response.text}")
        
        return {
            "images": image_paths,
            "model": model,
            "prompt": prompt,
            "size": size,
            "provider": "huggingface"
        }
    
    def edit_image(
        self,
        image_path: str,
        prompt: str,
        mask_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Edit an existing image using DALL-E 2
        
        Args:
            image_path: Path to original image
            prompt: Description of desired edit
            mask_path: Optional path to mask (transparent areas will be edited)
            
        Returns:
            Dict with edited image path
        """
        
        if "openai" not in self.providers:
            raise ValueError("Image editing requires OpenAI API key")
        
        with open(image_path, "rb") as image_file:
            kwargs = {
                "image": image_file,
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024"
            }
            
            if mask_path:
                with open(mask_path, "rb") as mask_file:
                    kwargs["mask"] = mask_file
                    response = self.providers["openai"].images.edit(**kwargs)
            else:
                response = self.providers["openai"].images.edit(**kwargs)
        
        # Download and save edited image
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        img = Image.open(BytesIO(image_response.content))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"edited_{timestamp}.png"
        filepath = self.output_dir / filename
        img.save(filepath)
        
        logger.info(f"✅ Edited image saved: {filepath}")
        
        return {
            "image": str(filepath),
            "prompt": prompt,
            "provider": "openai"
        }
    
    def create_variation(
        self,
        image_path: str,
        n: int = 1
    ) -> Dict[str, Any]:
        """
        Create variations of an existing image using DALL-E 2
        
        Args:
            image_path: Path to original image
            n: Number of variations to generate
            
        Returns:
            Dict with variation image paths
        """
        
        if "openai" not in self.providers:
            raise ValueError("Image variations require OpenAI API key")
        
        with open(image_path, "rb") as image_file:
            response = self.providers["openai"].images.create_variation(
                image=image_file,
                n=n,
                size="1024x1024"
            )
        
        # Download and save variations
        image_paths = []
        for i, image_data in enumerate(response.data):
            image_response = requests.get(image_data.url)
            img = Image.open(BytesIO(image_response.content))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"variation_{timestamp}_{i}.png"
            filepath = self.output_dir / filename
            img.save(filepath)
            image_paths.append(str(filepath))
        
        logger.info(f"✅ Generated {len(image_paths)} variations")
        
        return {
            "images": image_paths,
            "provider": "openai"
        }


# Example usage
if __name__ == "__main__":
    generator = ImageGenerator()
    
    print("🎨 Available models:", generator.get_available_models())
    
    if generator.get_available_models():
        result = generator.generate(
            prompt="A serene landscape with mountains and a lake at sunset",
            size="1024x1024"
        )
        print(f"\n✨ Generated images ({result['model']}):")
        for img_path in result["images"]:
            print(f"  📁 {img_path}")

