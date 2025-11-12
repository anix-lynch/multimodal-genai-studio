"""
Multimodal GenAI Studio - Main Application
A comprehensive multimodal AI application with Gradio UI
"""

import gradio as gr
import logging
from pathlib import Path

from config import (
    APIKeys, 
    ModelConfig, 
    ServerConfig, 
    get_available_features,
    OUTPUT_DIR
)
from src.text.generator import TextGenerator
from src.image.generator import ImageGenerator
from src.audio.transcriber import AudioTranscriber
from src.audio.synthesizer import TextToSpeech
from src.multimodal.pipeline import MultimodalPipeline

logger = logging.getLogger(__name__)

# Initialize generators
text_gen = TextGenerator()
image_gen = ImageGenerator()
audio_trans = AudioTranscriber()
tts = TextToSpeech()
pipeline = MultimodalPipeline()

# Check available features
features, api_status = get_available_features()


# ==================== Text Generation Tab ====================

def generate_text_ui(prompt, model, max_tokens, temperature, system_prompt):
    """Text generation interface"""
    try:
        result = text_gen.generate(
            prompt=prompt,
            model=model,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            system_prompt=system_prompt if system_prompt else None
        )
        
        output = f"**Generated Text:**\n\n{result['text']}\n\n"
        output += f"---\n\n"
        output += f"📊 **Model:** {result['model']}\n"
        output += f"📊 **Tokens:** {result['tokens']}\n"
        output += f"📊 **Provider:** {result['provider']}"
        
        return output
        
    except Exception as e:
        logger.error(f"Text generation error: {e}")
        return f"❌ Error: {str(e)}"


# ==================== Image Generation Tab ====================

def generate_image_ui(prompt, model, size, quality, style, n):
    """Image generation interface"""
    try:
        result = image_gen.generate(
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
            style=style,
            n=int(n)
        )
        
        info = f"✅ Generated {len(result['images'])} image(s)\n\n"
        info += f"**Prompt:** {result['prompt']}\n"
        info += f"**Model:** {result['model']}\n"
        info += f"**Size:** {result['size']}\n"
        
        if result.get('revised_prompt'):
            info += f"\n**Revised Prompt:** {result['revised_prompt']}\n"
        
        # Return first image for display, info text, and all image paths
        return result['images'][0], info, result['images']
        
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        return None, f"❌ Error: {str(e)}", []


# ==================== Audio Transcription Tab ====================

def transcribe_audio_ui(audio_file, model, language, task):
    """Audio transcription interface"""
    try:
        if audio_file is None:
            return "⚠️ Please upload an audio file"
        
        result = audio_trans.transcribe(
            audio_path=audio_file,
            model=model,
            language=language if language else None,
            task=task
        )
        
        output = f"**Transcription:**\n\n{result['text']}\n\n"
        output += f"---\n\n"
        output += f"📊 **Model:** {result['model']}\n"
        output += f"📊 **Language:** {result['language']}\n"
        output += f"📊 **Provider:** {result['provider']}\n"
        output += f"📁 **Saved to:** {result['output_file']}"
        
        return output
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return f"❌ Error: {str(e)}"


# ==================== Text-to-Speech Tab ====================

def synthesize_speech_ui(text, model, voice, language, speed):
    """Text-to-speech interface"""
    try:
        if not text.strip():
            return None, "⚠️ Please enter some text"
        
        result = tts.synthesize(
            text=text,
            model=model,
            voice=voice,
            language=language,
            speed=float(speed)
        )
        
        info = f"✅ Speech synthesized\n\n"
        info += f"**Model:** {result['model']}\n"
        
        if 'voice' in result:
            info += f"**Voice:** {result['voice']}\n"
        if 'language' in result:
            info += f"**Language:** {result['language']}\n"
        
        info += f"**Provider:** {result['provider']}\n"
        info += f"📁 **Saved to:** {result['audio_path']}"
        
        return result['audio_path'], info
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None, f"❌ Error: {str(e)}"


# ==================== Multimodal Pipelines Tab ====================

def story_to_multimedia_ui(story, generate_images, generate_audio, image_model, tts_model, tts_voice):
    """Story to multimedia pipeline"""
    try:
        if not story.strip():
            return None, None, "⚠️ Please enter a story"
        
        result = pipeline.story_to_multimedia(
            story=story,
            generate_images=generate_images,
            generate_audio=generate_audio,
            image_model=image_model,
            tts_model=tts_model,
            tts_voice=tts_voice
        )
        
        # Format output
        info = f"✅ Multimedia story created\n\n"
        
        if result['images']:
            info += f"**Images:** {len(result['images'])} scenes generated\n"
            image_display = [img['path'] for img in result['images']]
        else:
            image_display = None
        
        if result['audio_path']:
            info += f"**Audio:** Narration generated\n"
            audio_display = result['audio_path']
        else:
            audio_display = None
        
        info += f"\n📊 **Story length:** {len(story)} characters"
        
        return image_display, audio_display, info
        
    except Exception as e:
        logger.error(f"Multimedia pipeline error: {e}")
        return None, None, f"❌ Error: {str(e)}"


def audio_to_blog_ui(audio_file, transcribe_model, text_model, generate_image, image_model):
    """Audio to blog post pipeline"""
    try:
        if audio_file is None:
            return "", None, "⚠️ Please upload an audio file"
        
        result = pipeline.audio_to_blog_post(
            audio_path=audio_file,
            transcribe_model=transcribe_model,
            text_model=text_model,
            generate_image=generate_image,
            image_model=image_model
        )
        
        info = f"✅ Blog post created\n\n"
        info += f"**Transcription:** {len(result['transcription'])} characters\n"
        info += f"**Blog post:** Saved to {result['blog_file']}\n"
        
        if result['featured_image']:
            info += f"**Featured image:** Generated\n"
        
        return result['blog_post'], result['featured_image'], info
        
    except Exception as e:
        logger.error(f"Blog pipeline error: {e}")
        return "", None, f"❌ Error: {str(e)}"


# ==================== Gradio Interface ====================

def create_interface():
    """Create the Gradio interface"""
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        max-width: 1200px;
        margin: auto;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as app:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🎨 Multimodal GenAI Studio</h1>
            <p>Professional AI-powered content generation across text, image, and audio</p>
        </div>
        """)
        
        # API Status
        with gr.Accordion("📊 System Status", open=False):
            status_text = "**Available APIs:**\n\n"
            for api, status in api_status.items():
                status_text += f"- {api.capitalize()}: {'✅ Active' if status else '❌ Not configured'}\n"
            
            status_text += "\n**Available Features:**\n\n"
            for feature, available in features.items():
                status_text += f"- {feature.replace('_', ' ').title()}: {'✅ Available' if available else '❌ Unavailable'}\n"
            
            gr.Markdown(status_text)
        
        # Main Tabs
        with gr.Tabs():
            
            # ===== Text Generation Tab =====
            with gr.Tab("📝 Text Generation"):
                gr.Markdown("Generate text using multiple LLM providers")
                
                with gr.Row():
                    with gr.Column():
                        text_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your prompt here...",
                            lines=5
                        )
                        text_system = gr.Textbox(
                            label="System Prompt (Optional)",
                            placeholder="Set the AI's role and behavior...",
                            lines=2
                        )
                        
                        with gr.Row():
                            text_model = gr.Dropdown(
                                choices=text_gen.get_available_models(),
                                label="Model",
                                value=text_gen.get_available_models()[0] if text_gen.get_available_models() else None
                            )
                            text_max_tokens = gr.Slider(
                                minimum=100,
                                maximum=8192,
                                value=2048,
                                step=100,
                                label="Max Tokens"
                            )
                        
                        text_temperature = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.7,
                            step=0.1,
                            label="Temperature"
                        )
                        
                        text_generate_btn = gr.Button("Generate Text", variant="primary")
                    
                    with gr.Column():
                        text_output = gr.Markdown(label="Generated Text")
                
                text_generate_btn.click(
                    fn=generate_text_ui,
                    inputs=[text_prompt, text_model, text_max_tokens, text_temperature, text_system],
                    outputs=[text_output]
                )
            
            # ===== Image Generation Tab =====
            with gr.Tab("🎨 Image Generation"):
                gr.Markdown("Generate images using DALL-E or Stable Diffusion")
                
                with gr.Row():
                    with gr.Column():
                        image_prompt = gr.Textbox(
                            label="Image Prompt",
                            placeholder="Describe the image you want to generate...",
                            lines=4
                        )
                        
                        with gr.Row():
                            image_model = gr.Dropdown(
                                choices=image_gen.get_available_models(),
                                label="Model",
                                value=image_gen.get_available_models()[0] if image_gen.get_available_models() else None
                            )
                            image_size = gr.Dropdown(
                                choices=["1024x1024", "1792x1024", "1024x1792"],
                                label="Size",
                                value="1024x1024"
                            )
                        
                        with gr.Row():
                            image_quality = gr.Radio(
                                choices=["standard", "hd"],
                                label="Quality",
                                value="standard"
                            )
                            image_style = gr.Radio(
                                choices=["vivid", "natural"],
                                label="Style",
                                value="vivid"
                            )
                        
                        image_n = gr.Slider(
                            minimum=1,
                            maximum=4,
                            value=1,
                            step=1,
                            label="Number of Images"
                        )
                        
                        image_generate_btn = gr.Button("Generate Image", variant="primary")
                    
                    with gr.Column():
                        image_output = gr.Image(label="Generated Image")
                        image_info = gr.Markdown()
                        image_gallery = gr.Gallery(label="All Generated Images", columns=2)
                
                image_generate_btn.click(
                    fn=generate_image_ui,
                    inputs=[image_prompt, image_model, image_size, image_quality, image_style, image_n],
                    outputs=[image_output, image_info, image_gallery]
                )
            
            # ===== Audio Transcription Tab =====
            with gr.Tab("🎤 Audio Transcription"):
                gr.Markdown("Transcribe audio to text using Whisper")
                
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            label="Upload Audio",
                            type="filepath"
                        )
                        
                        with gr.Row():
                            trans_model = gr.Dropdown(
                                choices=audio_trans.get_available_models(),
                                label="Model",
                                value=audio_trans.get_available_models()[0] if audio_trans.get_available_models() else None
                            )
                            trans_language = gr.Textbox(
                                label="Language (Optional)",
                                placeholder="e.g., en, ja, es",
                                value=""
                            )
                        
                        trans_task = gr.Radio(
                            choices=["transcribe", "translate"],
                            label="Task",
                            value="transcribe"
                        )
                        
                        trans_btn = gr.Button("Transcribe", variant="primary")
                    
                    with gr.Column():
                        trans_output = gr.Markdown(label="Transcription")
                
                trans_btn.click(
                    fn=transcribe_audio_ui,
                    inputs=[audio_input, trans_model, trans_language, trans_task],
                    outputs=[trans_output]
                )
            
            # ===== Text-to-Speech Tab =====
            with gr.Tab("🔊 Text-to-Speech"):
                gr.Markdown("Convert text to speech using OpenAI TTS or gTTS")
                
                with gr.Row():
                    with gr.Column():
                        tts_text = gr.Textbox(
                            label="Text to Synthesize",
                            placeholder="Enter text to convert to speech...",
                            lines=5
                        )
                        
                        with gr.Row():
                            tts_model = gr.Dropdown(
                                choices=tts.get_available_models(),
                                label="Model",
                                value=tts.get_available_models()[0] if tts.get_available_models() else None
                            )
                        tts_voice = gr.Dropdown(
                            choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                            label="Voice",
                            value="alloy"
                        )
                        
                        with gr.Row():
                            tts_language = gr.Textbox(
                                label="Language (for gTTS)",
                                value="en"
                            )
                            tts_speed = gr.Slider(
                                minimum=0.25,
                                maximum=4.0,
                                value=1.0,
                                step=0.25,
                                label="Speed"
                            )
                        
                        tts_btn = gr.Button("Generate Speech", variant="primary")
                    
                    with gr.Column():
                        tts_audio = gr.Audio(label="Generated Speech")
                        tts_info = gr.Markdown()
                
                tts_btn.click(
                    fn=synthesize_speech_ui,
                    inputs=[tts_text, tts_model, tts_voice, tts_language, tts_speed],
                    outputs=[tts_audio, tts_info]
                )
            
            # ===== Multimodal Pipelines Tab =====
            with gr.Tab("🎭 Multimodal Pipelines"):
                gr.Markdown("Advanced workflows combining multiple modalities")
                
                with gr.Tabs():
                    
                    # Story to Multimedia
                    with gr.Tab("📚 Story to Multimedia"):
                        with gr.Row():
                            with gr.Column():
                                story_text = gr.Textbox(
                                    label="Story",
                                    placeholder="Enter your story...",
                                    lines=10
                                )
                                
                                story_gen_images = gr.Checkbox(
                                    label="Generate Scene Images",
                                    value=True
                                )
                                story_gen_audio = gr.Checkbox(
                                    label="Generate Narration",
                                    value=True
                                )
                                
                                with gr.Row():
                                    story_img_model = gr.Dropdown(
                                        choices=image_gen.get_available_models(),
                                        label="Image Model",
                                        value=image_gen.get_available_models()[0] if image_gen.get_available_models() else None
                                    )
                                    story_tts_model = gr.Dropdown(
                                        choices=tts.get_available_models(),
                                        label="TTS Model",
                                        value=tts.get_available_models()[0] if tts.get_available_models() else None
                                    )
                                
                                story_tts_voice = gr.Dropdown(
                                    choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                                    label="Voice",
                                    value="nova"
                                )
                                
                                story_btn = gr.Button("Create Multimedia Story", variant="primary")
                            
                            with gr.Column():
                                story_images = gr.Gallery(label="Scene Images", columns=2)
                                story_audio = gr.Audio(label="Narration")
                                story_info = gr.Markdown()
                        
                        story_btn.click(
                            fn=story_to_multimedia_ui,
                            inputs=[story_text, story_gen_images, story_gen_audio, story_img_model, story_tts_model, story_tts_voice],
                            outputs=[story_images, story_audio, story_info]
                        )
                    
                    # Audio to Blog
                    with gr.Tab("🎙️ Audio to Blog Post"):
                        with gr.Row():
                            with gr.Column():
                                blog_audio = gr.Audio(
                                    label="Upload Audio",
                                    type="filepath"
                                )
                                
                                with gr.Row():
                                    blog_trans_model = gr.Dropdown(
                                        choices=audio_trans.get_available_models(),
                                        label="Transcription Model",
                                        value=audio_trans.get_available_models()[0] if audio_trans.get_available_models() else None
                                    )
                                    blog_text_model = gr.Dropdown(
                                        choices=text_gen.get_available_models(),
                                        label="Text Model",
                                        value=text_gen.get_available_models()[0] if text_gen.get_available_models() else None
                                    )
                                
                                blog_gen_image = gr.Checkbox(
                                    label="Generate Featured Image",
                                    value=True
                                )
                                blog_img_model = gr.Dropdown(
                                    choices=image_gen.get_available_models(),
                                    label="Image Model",
                                    value=image_gen.get_available_models()[0] if image_gen.get_available_models() else None
                                )
                                
                                blog_btn = gr.Button("Create Blog Post", variant="primary")
                            
                            with gr.Column():
                                blog_post = gr.Markdown(label="Blog Post")
                                blog_image = gr.Image(label="Featured Image")
                                blog_info = gr.Markdown()
                        
                        blog_btn.click(
                            fn=audio_to_blog_ui,
                            inputs=[blog_audio, blog_trans_model, blog_text_model, blog_gen_image, blog_img_model],
                            outputs=[blog_post, blog_image, blog_info]
                        )
        
        # Footer
        gr.Markdown("""
        ---
        **Multimodal GenAI Studio** | Built with Gradio | Showcasing IBM's Build Multimodal Generative AI Applications certification
        """)
    
    return app


# ==================== Main ====================

if __name__ == "__main__":
    logger.info("🚀 Starting Multimodal GenAI Studio...")
    
    # Check API keys
    features, api_status = get_available_features()
    
    print("\n" + "="*50)
    print("🎨 Multimodal GenAI Studio")
    print("="*50)
    
    print("\n📊 API Status:")
    for api, status in api_status.items():
        print(f"  {api.capitalize()}: {'✅' if status else '❌'}")
    
    print("\n🎯 Available Features:")
    for feature, available in features.items():
        print(f"  {feature.replace('_', ' ').title()}: {'✅' if available else '❌'}")
    
    if not any(api_status.values()):
        print("\n⚠️  WARNING: No API keys configured!")
        print("   Set API keys in .env file to enable features")
    
    print(f"\n🌐 Starting server on {ServerConfig.HOST}:{ServerConfig.PORT}")
    print("="*50 + "\n")
    
    # Create and launch app
    app = create_interface()
    app.launch()  # HF Spaces handles all config automatically

