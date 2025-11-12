"""
Multimodal GenAI Studio - Streamlit Version
Simplified version for Streamlit Cloud deployment
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set page config
st.set_page_config(
    page_title="Multimodal GenAI Studio",
    page_icon="🎨",
    layout="wide"
)

# Load secrets (works with both local and cloud)
try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
except:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Set environment variable
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize generators
@st.cache_resource
def init_generators():
    try:
        from src.text.generator import TextGenerator
        from src.audio.synthesizer import TextToSpeech
        
        text_gen = TextGenerator()
        tts = TextToSpeech()
        return text_gen, tts
    except Exception as e:
        st.error(f"Import error: {e}")
        return None, None

# Header
st.markdown("# 🎨 Multimodal GenAI Studio")
st.markdown("Professional AI-powered content generation")

if not GOOGLE_API_KEY:
    st.error("⚠️ No API key configured. Add GOOGLE_API_KEY in Streamlit Cloud secrets.")
    st.stop()

text_gen, tts = init_generators()

if text_gen is None or tts is None:
    st.error("Failed to initialize generators. Check logs.")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Text Generation", "🔊 Text-to-Speech", "ℹ️ About"])

with tab1:
    st.header("Text Generation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area("Prompt", placeholder="Enter your prompt here...", height=150)
    
    with col2:
        models = text_gen.get_available_models()
        if models:
            model = st.selectbox("Model", models, index=0)
        else:
            st.error("No models available")
            st.stop()
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        max_tokens = st.slider("Max Tokens", 100, 4096, 2048, 100)
    
    if st.button("Generate Text", type="primary"):
        if prompt:
            with st.spinner("Generating..."):
                try:
                    result = text_gen.generate(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    st.markdown("### Generated Text")
                    st.write(result['text'])
                    
                    st.caption(f"Model: {result['model']} | Tokens: {result['tokens']} | Provider: {result['provider']}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a prompt")

with tab2:
    st.header("Text-to-Speech")
    
    text = st.text_area("Text to Synthesize", placeholder="Enter text to convert to speech...", height=150)
    
    col1, col2 = st.columns(2)
    with col1:
        tts_models = tts.get_available_models()
        if tts_models:
            tts_model = st.selectbox("Model", tts_models, index=0)
        else:
            st.error("No TTS models available")
            st.stop()
    
    with col2:
        language = st.text_input("Language (for gTTS)", value="en")
    
    if st.button("Generate Speech", type="primary"):
        if text:
            with st.spinner("Generating speech..."):
                try:
                    result = tts.synthesize(
                        text=text,
                        model=tts_model,
                        language=language
                    )
                    
                    st.success("✅ Speech generated!")
                    
                    # Play audio
                    with open(result['audio_path'], 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/mp3')
                    
                    st.caption(f"Model: {result['model']} | Provider: {result['provider']}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter text")

with tab3:
    st.header("About")
    st.markdown("""
    ### Features
    
    **Current (FREE tier):**
    - ✅ Text Generation (Google Gemini)
    - ✅ Text-to-Speech (gTTS)
    
    **With additional API keys:**
    - 🎨 Image Generation (DALL-E, Stable Diffusion)
    - 🎤 Audio Transcription (Whisper)
    - 🎭 Multimodal Pipelines
    
    ### Cost
    - **FREE** with just Gemini API key
    - Gemini: 60 requests/min free
    - gTTS: Unlimited free
    
    ### GitHub
    [View Source](https://github.com/anixlynch/coursera-portfolio-projects)
    
    ### Certifications
    Showcases skills from:
    - Build Multimodal Generative AI Applications (IBM)
    - Python for Data Science, AI & Development (IBM)
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Powered by Google Gemini")

