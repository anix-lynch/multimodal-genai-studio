"""
Multimodal GenAI Studio - Streamlit Version
Showcases: Build Multimodal Generative AI Applications (IBM)
Perfect for Streamlit Cloud deployment
"""

import streamlit as st
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules
from src.text.generator import TextGenerator
from src.image.generator import ImageGenerator
from src.audio.transcriber import AudioTranscriber
from src.audio.synthesizer import TextToSpeech
from src.multimodal.pipeline import MultimodalPipeline

# Configure page
st.set_page_config(
    page_title="Multimodal GenAI Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'text_gen' not in st.session_state:
    st.session_state.text_gen = TextGenerator()
if 'image_gen' not in st.session_state:
    st.session_state.image_gen = ImageGenerator()
if 'audio_trans' not in st.session_state:
    st.session_state.audio_trans = AudioTranscriber()
if 'tts' not in st.session_state:
    st.session_state.tts = TextToSpeech()
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = MultimodalPipeline()

# Header
st.title("🎨 Multimodal GenAI Studio")
st.markdown("Create and combine text, images, and audio using various AI models.")
st.markdown("**🎓 IBM Coursera Certification:** Build Multimodal Generative AI Applications")

# Sidebar for API status
with st.sidebar:
    st.header("🔑 API Status")

    # Check API keys
    gemini_key = bool(os.getenv("GEMINI_API_KEY", ""))
    openai_key = bool(os.getenv("OPENAI_API_KEY", ""))
    hf_token = bool(os.getenv("HF_TOKEN", ""))

    st.write(f"**Gemini:** {'✅' if gemini_key else '❌'}")
    st.write(f"**OpenAI:** {'✅' if openai_key else '❌'}")
    st.write(f"**HuggingFace:** {'✅' if hf_token else '❌'}")

    if not any([gemini_key, openai_key, hf_token]):
        st.warning("⚠️ No API keys configured. Some features may not work.")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Text Generation", "🖼️ Image Generation", "🎤 Audio Processing", "🎯 Multimodal", "ℹ️ About"])

with tab1:
    st.header("📝 Text Generation")

    col1, col2 = st.columns([2, 1])

    with col1:
        prompt = st.text_area(
            "Enter your prompt:",
            placeholder="Write a creative story about...",
            height=150
        )

        model = st.selectbox(
            "Model:",
            ["gemini-1.5-flash", "gpt-4o-mini", "claude-3-haiku"],
            index=0
        )

        max_tokens = st.slider("Max tokens:", 100, 2000, 500)

    with col2:
        temperature = st.slider("Creativity:", 0.0, 2.0, 0.7)
        system_prompt = st.text_area("System prompt (optional):", height=100)

    if st.button("Generate Text", type="primary", use_container_width=True):
        if prompt.strip():
            with st.spinner("🤖 Generating text..."):
                try:
                    result = st.session_state.text_gen.generate(
                        prompt=prompt,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system_prompt=system_prompt if system_prompt else None
                    )

                    st.success("✅ Generated!")
                    st.markdown("### Result:")
                    st.markdown(result['text'])

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please enter a prompt.")

with tab2:
    st.header("🖼️ Image Generation")

    col1, col2 = st.columns([2, 1])

    with col1:
        image_prompt = st.text_area(
            "Describe the image:",
            placeholder="A beautiful sunset over mountains...",
            height=100
        )

        image_model = st.selectbox(
            "Model:",
            ["dall-e-3", "stable-diffusion-xl"],
            index=0
        )

    with col2:
        if image_model == "dall-e-3":
            image_size = st.selectbox("Size:", ["1024x1024", "1792x1024", "1024x1792"])
        else:
            image_size = st.selectbox("Size:", ["512x512", "1024x1024"])

    if st.button("Generate Image", type="primary", use_container_width=True):
        if image_prompt.strip():
            with st.spinner("🎨 Generating image..."):
                try:
                    result = st.session_state.image_gen.generate(
                        prompt=image_prompt,
                        model=image_model,
                        size=image_size
                    )

                    st.success("✅ Generated!")
                    if 'url' in result:
                        st.image(result['url'], caption=image_prompt)
                    elif 'image' in result:
                        st.image(result['image'], caption=image_prompt)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please describe the image.")

with tab3:
    st.header("🎤 Audio Processing")

    audio_tab1, audio_tab2 = st.tabs(["🎧 Transcription", "🔊 Text-to-Speech"])

    with audio_tab1:
        st.subheader("Speech to Text")
        audio_file = st.file_uploader("Upload audio file", type=['mp3', 'wav', 'm4a'])

        if st.button("Transcribe Audio") and audio_file:
            with st.spinner("🎧 Transcribing..."):
                try:
                    # Save temp file
                    with open(f"temp_audio.{audio_file.type.split('/')[-1]}", "wb") as f:
                        f.write(audio_file.getbuffer())

                    result = st.session_state.audio_trans.transcribe(f"temp_audio.{audio_file.type.split('/')[-1]}")

                    st.success("✅ Transcribed!")
                    st.markdown("### Transcription:")
                    st.write(result['text'])

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    with audio_tab2:
        st.subheader("Text to Speech")
        tts_text = st.text_area("Enter text to convert to speech:", height=100)
        tts_voice = st.selectbox("Voice:", ["alloy", "echo", "fable", "onyx"])

        if st.button("Generate Speech") and tts_text.strip():
            with st.spinner("🔊 Generating speech..."):
                try:
                    result = st.session_state.tts.synthesize(
                        text=tts_text,
                        voice=tts_voice
                    )

                    st.success("✅ Generated!")
                    if 'audio_url' in result:
                        st.audio(result['audio_url'])
                    elif 'audio' in result:
                        st.audio(result['audio'])

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

with tab4:
    st.header("🎯 Multimodal Pipeline")

    st.markdown("Combine multiple modalities for creative content generation.")

    col1, col2 = st.columns(2)

    with col1:
        multimodal_prompt = st.text_area(
            "Creative concept:",
            placeholder="A story about a robot learning to paint...",
            height=100
        )

    with col2:
        content_type = st.selectbox(
            "Output type:",
            ["Story with image", "Blog post with audio", "Presentation slides"]
        )

    if st.button("Create Multimodal Content", type="primary", use_container_width=True):
        if multimodal_prompt.strip():
            with st.spinner("🎨 Creating multimodal content..."):
                try:
                    result = st.session_state.pipeline.create_multimodal_content(
                        prompt=multimodal_prompt,
                        content_type=content_type
                    )

                    st.success("✅ Created!")

                    # Display results based on type
                    if content_type == "Story with image":
                        st.markdown("### 📖 Story:")
                        st.write(result.get('story', ''))
                        if 'image' in result:
                            st.image(result['image'])
                    elif content_type == "Blog post with audio":
                        st.markdown("### 📝 Blog Post:")
                        st.write(result.get('blog_post', ''))
                        if 'audio' in result:
                            st.audio(result['audio'])
                    elif content_type == "Presentation slides":
                        st.markdown("### 📊 Presentation:")
                        st.write(result.get('slides', ''))

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please enter a creative concept.")

with tab5:
    st.header("ℹ️ About This Project")

    st.markdown("""
    This Multimodal GenAI Studio showcases skills from the **IBM Coursera certification: Build Multimodal Generative AI Applications**.

    ### 🎓 IBM Certification Skills Demonstrated
    - **Text Generation:** Using Gemini, GPT, and Claude models
    - **Image Generation:** DALL-E 3 and Stable Diffusion integration
    - **Audio Processing:** Speech-to-text transcription
    - **Text-to-Speech:** Voice synthesis capabilities
    - **Multimodal Pipelines:** Combining multiple AI modalities

    ### 🛠️ Tech Stack
    - **Frontend:** Streamlit (interactive web UI)
    - **AI Models:** Google Gemini, OpenAI GPT, Anthropic Claude
    - **Image Gen:** DALL-E 3, Stable Diffusion via HuggingFace
    - **Audio:** OpenAI Whisper, TTS-1
    - **Backend:** Python with custom AI pipeline classes

    ### 🚀 Features
    - Text generation with multiple models
    - AI-powered image creation
    - Audio transcription and synthesis
    - Multimodal content creation
    - Real-time processing with progress indicators

    ---
    **Author:** Anix Lynch | [Portfolio](https://gozeroshot.dev) | [GitHub](https://github.com/anix-lynch)
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ showcasing IBM Coursera Multimodal GenAI certification")
