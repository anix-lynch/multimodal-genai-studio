"""
Multimodal Pipeline Module
Combines text, image, and audio generation in unified workflows
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import json

from src.text.generator import TextGenerator
from src.image.generator import ImageGenerator
from src.audio.transcriber import AudioTranscriber
from src.audio.synthesizer import TextToSpeech
from config import OUTPUT_DIR

logger = logging.getLogger(__name__)


class MultimodalPipeline:
    """
    Orchestrates multimodal workflows combining text, image, and audio
    """
    
    def __init__(self):
        """Initialize all generation modules"""
        self.text_gen = TextGenerator()
        self.image_gen = ImageGenerator()
        self.audio_trans = AudioTranscriber()
        self.tts = TextToSpeech()
        
        self.output_dir = OUTPUT_DIR / "multimodal"
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("✅ Multimodal pipeline initialized")
    
    def text_to_speech_to_text(
        self,
        original_text: str,
        tts_model: str = "gtts",
        transcribe_model: str = "whisper-1",
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Pipeline: Text -> Speech -> Text (roundtrip test)
        
        Args:
            original_text: Original text
            tts_model: TTS model to use
            transcribe_model: Transcription model to use
            language: Language code
            
        Returns:
            Dict with original text, audio path, transcribed text
        """
        
        logger.info("🔄 Starting text->speech->text pipeline")
        
        # Step 1: Text to Speech
        tts_result = self.tts.synthesize(
            text=original_text,
            model=tts_model,
            language=language
        )
        logger.info(f"✅ Speech generated: {tts_result['audio_path']}")
        
        # Step 2: Speech to Text
        transcribe_result = self.audio_trans.transcribe(
            audio_path=tts_result['audio_path'],
            model=transcribe_model,
            language=language
        )
        logger.info(f"✅ Speech transcribed")
        
        return {
            "original_text": original_text,
            "audio_path": tts_result['audio_path'],
            "transcribed_text": transcribe_result['text'],
            "tts_model": tts_model,
            "transcribe_model": transcribe_model
        }
    
    def text_to_image_to_caption(
        self,
        prompt: str,
        image_model: str = "dall-e-3",
        caption_model: str = "gemini-1.5-flash"
    ) -> Dict[str, Any]:
        """
        Pipeline: Text -> Image -> Caption
        Generate image from text, then describe it
        
        Args:
            prompt: Image generation prompt
            image_model: Image generation model
            caption_model: Text model for captioning
            
        Returns:
            Dict with prompt, image path, generated caption
        """
        
        logger.info("🔄 Starting text->image->caption pipeline")
        
        # Step 1: Generate Image
        image_result = self.image_gen.generate(
            prompt=prompt,
            model=image_model,
            n=1
        )
        image_path = image_result['images'][0]
        logger.info(f"✅ Image generated: {image_path}")
        
        # Step 2: Caption the Image
        caption_prompt = f"Describe this image in detail. The image was generated with the prompt: '{prompt}'. Analyze how well the image matches the prompt and describe what you see."
        
        caption_result = self.text_gen.generate(
            prompt=caption_prompt,
            model=caption_model
        )
        logger.info(f"✅ Caption generated")
        
        return {
            "original_prompt": prompt,
            "image_path": image_path,
            "caption": caption_result['text'],
            "image_model": image_model,
            "caption_model": caption_model
        }
    
    def story_to_multimedia(
        self,
        story: str,
        generate_images: bool = True,
        generate_audio: bool = True,
        image_model: str = "dall-e-3",
        tts_model: str = "tts-1",
        tts_voice: str = "nova"
    ) -> Dict[str, Any]:
        """
        Pipeline: Story -> Images + Audio narration
        Create multimedia content from a story
        
        Args:
            story: Story text
            generate_images: Whether to generate scene images
            generate_audio: Whether to generate narration
            image_model: Image generation model
            tts_model: TTS model
            tts_voice: TTS voice
            
        Returns:
            Dict with story, images, audio path
        """
        
        logger.info("🔄 Starting story->multimedia pipeline")
        
        results = {
            "story": story,
            "images": [],
            "audio_path": None
        }
        
        # Step 1: Extract scenes for image generation
        if generate_images:
            logger.info("Extracting scenes from story...")
            
            scene_prompt = f"""Analyze this story and extract 3-4 key visual scenes that would make good illustrations.
For each scene, provide a detailed image generation prompt.

Story:
{story}

Return ONLY a JSON array of image prompts, nothing else. Format:
["prompt 1", "prompt 2", "prompt 3"]
"""
            
            scene_result = self.text_gen.generate(
                prompt=scene_prompt,
                model="gemini-1.5-flash",
                temperature=0.7
            )
            
            # Parse scene prompts
            try:
                import re
                # Extract JSON array from response
                json_match = re.search(r'\[.*\]', scene_result['text'], re.DOTALL)
                if json_match:
                    scene_prompts = json.loads(json_match.group())
                else:
                    # Fallback: split by lines
                    scene_prompts = [line.strip(' -"\'') for line in scene_result['text'].split('\n') if line.strip()][:3]
                
                logger.info(f"Extracted {len(scene_prompts)} scenes")
                
                # Generate images for each scene
                for i, scene_prompt in enumerate(scene_prompts[:3]):  # Limit to 3 images
                    logger.info(f"Generating image {i+1}/{len(scene_prompts[:3])}")
                    img_result = self.image_gen.generate(
                        prompt=scene_prompt,
                        model=image_model,
                        n=1
                    )
                    results['images'].append({
                        "prompt": scene_prompt,
                        "path": img_result['images'][0]
                    })
                
                logger.info(f"✅ Generated {len(results['images'])} scene images")
                
            except Exception as e:
                logger.error(f"Scene extraction failed: {e}")
        
        # Step 2: Generate audio narration
        if generate_audio:
            logger.info("Generating audio narration...")
            
            audio_result = self.tts.synthesize(
                text=story,
                model=tts_model,
                voice=tts_voice
            )
            results['audio_path'] = audio_result['audio_path']
            logger.info(f"✅ Narration generated: {results['audio_path']}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"multimedia_story_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Multimedia story saved: {output_file}")
        
        return results
    
    def audio_to_blog_post(
        self,
        audio_path: str,
        transcribe_model: str = "whisper-1",
        text_model: str = "gemini-1.5-flash",
        generate_image: bool = True,
        image_model: str = "dall-e-3"
    ) -> Dict[str, Any]:
        """
        Pipeline: Audio -> Text -> Blog Post + Featured Image
        Convert audio recording to a formatted blog post
        
        Args:
            audio_path: Path to audio file
            transcribe_model: Transcription model
            text_model: Text generation model
            generate_image: Whether to generate featured image
            image_model: Image generation model
            
        Returns:
            Dict with transcription, blog post, image
        """
        
        logger.info("🔄 Starting audio->blog pipeline")
        
        # Step 1: Transcribe audio
        transcribe_result = self.audio_trans.transcribe(
            audio_path=audio_path,
            model=transcribe_model
        )
        transcription = transcribe_result['text']
        logger.info(f"✅ Audio transcribed ({len(transcription)} chars)")
        
        # Step 2: Convert to blog post
        blog_prompt = f"""Convert this transcription into a well-formatted blog post.

Requirements:
- Add a compelling title
- Format with proper markdown (headers, paragraphs, lists)
- Fix any transcription errors
- Improve flow and readability
- Keep the original meaning and tone

Transcription:
{transcription}

Return ONLY the formatted blog post in markdown.
"""
        
        blog_result = self.text_gen.generate(
            prompt=blog_prompt,
            model=text_model,
            temperature=0.7
        )
        blog_post = blog_result['text']
        logger.info(f"✅ Blog post generated")
        
        # Step 3: Generate featured image
        featured_image = None
        if generate_image:
            # Extract title for image prompt
            lines = blog_post.split('\n')
            title = lines[0].strip('# ') if lines else "Blog post"
            
            image_prompt = f"Professional featured image for blog post titled: {title}. Modern, clean, high-quality illustration."
            
            image_result = self.image_gen.generate(
                prompt=image_prompt,
                model=image_model,
                n=1
            )
            featured_image = image_result['images'][0]
            logger.info(f"✅ Featured image generated: {featured_image}")
        
        # Save blog post
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blog_file = self.output_dir / f"blog_post_{timestamp}.md"
        blog_file.write_text(blog_post)
        
        logger.info(f"✅ Blog post saved: {blog_file}")
        
        return {
            "transcription": transcription,
            "blog_post": blog_post,
            "blog_file": str(blog_file),
            "featured_image": featured_image,
            "transcribe_model": transcribe_model,
            "text_model": text_model
        }
    
    def creative_chain(
        self,
        initial_prompt: str,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Pipeline: Creative chain across modalities
        Text -> Image -> Description -> New Image -> ...
        
        Args:
            initial_prompt: Starting prompt
            iterations: Number of creative iterations
            
        Returns:
            Dict with chain of prompts and images
        """
        
        logger.info(f"🔄 Starting creative chain ({iterations} iterations)")
        
        chain = []
        current_prompt = initial_prompt
        
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations}")
            
            # Generate image
            image_result = self.image_gen.generate(
                prompt=current_prompt,
                model="dall-e-3",
                n=1
            )
            image_path = image_result['images'][0]
            
            # Generate new prompt based on image
            if i < iterations - 1:  # Don't generate prompt on last iteration
                new_prompt_request = f"Describe this image in vivid detail, then suggest a creative variation or evolution of this concept for a new image. Be creative and unexpected."
                
                new_prompt_result = self.text_gen.generate(
                    prompt=new_prompt_request,
                    model="gemini-1.5-flash"
                )
                
                # Extract the variation idea
                next_prompt = new_prompt_result['text']
            else:
                next_prompt = None
            
            chain.append({
                "iteration": i + 1,
                "prompt": current_prompt,
                "image": image_path,
                "next_prompt": next_prompt
            })
            
            current_prompt = next_prompt
            logger.info(f"✅ Iteration {i+1} complete")
        
        # Save chain
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chain_file = self.output_dir / f"creative_chain_{timestamp}.json"
        
        with open(chain_file, 'w') as f:
            json.dump(chain, f, indent=2)
        
        logger.info(f"✅ Creative chain saved: {chain_file}")
        
        return {
            "initial_prompt": initial_prompt,
            "iterations": iterations,
            "chain": chain,
            "chain_file": str(chain_file)
        }


# Example usage
if __name__ == "__main__":
    pipeline = MultimodalPipeline()
    
    print("🎭 Multimodal Pipeline Ready")
    print("Available workflows:")
    print("  1. text_to_speech_to_text")
    print("  2. text_to_image_to_caption")
    print("  3. story_to_multimedia")
    print("  4. audio_to_blog_post")
    print("  5. creative_chain")

