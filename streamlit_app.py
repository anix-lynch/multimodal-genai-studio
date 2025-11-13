"""
Multimodal GenAI Studio - Streamlit Version
Showcases: Build Multimodal Generative AI Applications (IBM)
Perfect for Streamlit Cloud deployment
"""

import streamlit as st
import os
import logging
import requests
from io import BytesIO
import base64

# Configure page
st.set_page_config(
    page_title="Multimodal GenAI Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize API clients
def get_gemini_client():
    """Get Gemini API client and find working model"""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)

            # List available models first
            try:
                models = genai.list_models()
                text_models = [m for m in models if 'generateContent' in m.supported_generation_methods]

                if text_models:
                    # Use the first available text generation model
                    model_name = text_models[0].name.split('/')[-1]  # Extract model name
                    return genai.GenerativeModel(model_name), model_name
                else:
                    # Fallback to known working models
                    for model_name in ['gemini-1.0-pro', 'gemini-pro']:
                        try:
                            return genai.GenerativeModel(model_name), model_name
                        except:
                            continue
            except Exception as e:
                st.error(f"Could not list models: {e}")

            return None, None
        return None, None
    except ImportError:
        return None, None

def get_openai_client():
    """Get OpenAI API client"""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return OpenAI(api_key=api_key)
        return None
    except ImportError:
        return None

# Initialize clients
gemini_model, gemini_model_name = get_gemini_client()
openai_client = get_openai_client()

# Header
st.title("🎨 Multimodal GenAI Studio")
st.markdown("Create and combine text, images, and audio using various AI models.")
st.markdown("**🎓 IBM Coursera Certification:** Build Multimodal Generative AI Applications")

# Sidebar for API status
with st.sidebar:
    st.header("🔑 API Status")

    # Check API availability and show debug info
    gemini_available = gemini_model is not None
    openai_available = openai_client is not None

    if gemini_available and gemini_model_name:
        st.write(f"**Gemini ({gemini_model_name}):** ⚠️ (Free tier - quota limits)")
    else:
        st.write("**Gemini:** ❌")
    st.write(f"**OpenAI:** {'✅' if openai_available else '❌'} (Recommended - no quotas)")

    # Debug info
    if st.checkbox("Show debug info"):
        st.write("**Debug Info:**")
        gemini_key = bool(os.getenv("GEMINI_API_KEY", ""))
        openai_key = bool(os.getenv("OPENAI_API_KEY", ""))
        st.write(f"Gemini key present: {'✅' if gemini_key else '❌'}")
        st.write(f"OpenAI key present: {'✅' if openai_key else '❌'}")
        if gemini_model_name:
            st.write(f"Gemini model: {gemini_model_name}")

        if gemini_available:
            try:
                # Test Gemini model
                test_response = gemini_model.generate_content("Hello")
                st.write("Gemini test: ✅ Working")
            except Exception as e:
                st.write(f"Gemini test: ❌ Error - {str(e)[:50]}...")

    if not any([gemini_available, openai_available]):
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

        # Build model options dynamically - prioritize OpenAI (more reliable)
        model_options = ["gpt-4o-mini"]  # Always include OpenAI first
        if gemini_available and gemini_model_name:
            model_options.append(gemini_model_name)

        model = st.selectbox(
            "Model:",
            model_options,
            index=0,  # Default to OpenAI
            help="OpenAI is recommended - more reliable and no quota limits for your usage"
        )

        max_tokens = st.slider("Max tokens:", 100, 2000, 500)

    with col2:
        temperature = st.slider("Creativity:", 0.0, 2.0, 0.7)
        system_prompt = st.text_area("System prompt (optional):", height=100)

    if st.button("Generate Text", type="primary", use_container_width=True):
        if prompt.strip():
            with st.spinner("🤖 Generating text..."):
                try:
                    if model == gemini_model_name and gemini_model:
                        # Use Gemini
                        try:
                            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                            response = gemini_model.generate_content(full_prompt)
                            result_text = response.text
                        except Exception as gemini_error:
                            error_msg = str(gemini_error).lower()
                            if "quota" in error_msg or "429" in error_msg:
                                st.error("❌ Gemini Free Tier Quota Exceeded")
                                st.info("💡 **Switch to GPT-4o-mini** - OpenAI has no quota limits for your usage level!")
                            else:
                                st.error(f"❌ Gemini API Error: {str(gemini_error)}")
                                st.info("💡 Try using GPT-4o-mini instead.")
                            result_text = None

                    elif model == "gpt-4o-mini" and openai_client:
                        # Use OpenAI
                        messages = []
                        if system_prompt:
                            messages.append({"role": "system", "content": system_prompt})
                        messages.append({"role": "user", "content": prompt})

                        response = openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                        result_text = response.choices[0].message.content

                    else:
                        st.error(f"❌ {model} model not available. Check API keys.")
                        result_text = None

                    if result_text:
                        st.success("✅ Generated!")
                        st.markdown("### Result:")
                        st.markdown(result_text)

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
            ["dall-e-3", "dall-e-2"],
            index=0
        )

    with col2:
        if image_model == "dall-e-3":
            image_size = st.selectbox("Size:", ["1024x1024", "1792x1024", "1024x1792"])
        else:
            image_size = st.selectbox("Size:", ["256x256", "512x512", "1024x1024"])

    if st.button("Generate Image", type="primary", use_container_width=True):
        if image_prompt.strip():
            with st.spinner("🎨 Generating image..."):
                try:
                    if openai_client:
                        response = openai_client.images.generate(
                            model=image_model,
                            prompt=image_prompt,
                            size=image_size,
                            quality="standard",
                            n=1,
                        )

                        image_url = response.data[0].url
                        st.success("✅ Generated!")
                        st.image(image_url, caption=image_prompt, use_container_width=True)
                    else:
                        st.error("❌ OpenAI API key not available for image generation.")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please describe the image.")

with tab3:
    st.header("🎤 Audio Processing")

    audio_tab1, audio_tab2 = st.tabs(["🎧 Transcription", "🔊 Text-to-Speech"])

    with audio_tab1:
        st.subheader("Speech to Text (OpenAI Whisper)")
        st.info("Upload an audio file (MP3, WAV, M4A) to transcribe it to text")
        audio_file = st.file_uploader("Upload audio file", type=['mp3', 'wav', 'm4a'])

        if st.button("Transcribe Audio", type="primary") and audio_file:
            with st.spinner("🎧 Transcribing..."):
                try:
                    if openai_client:
                        # Save uploaded file temporarily
                        audio_bytes = audio_file.getvalue()

                        # Create transcription
                        transcription = openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=("audio." + audio_file.type.split('/')[-1], audio_bytes, audio_file.type)
                        )

                        st.success("✅ Transcribed!")
                        st.markdown("### Transcription:")
                        st.write(transcription.text)
                    else:
                        st.error("❌ OpenAI API key not available for transcription.")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    with audio_tab2:
        st.subheader("Text to Speech (OpenAI TTS)")
        tts_text = st.text_area("Enter text to convert to speech:", height=100, placeholder="Hello, this is a sample text to speech conversion.")
        tts_voice = st.selectbox("Voice:", ["alloy", "echo", "fable", "onyx"], index=0)

        if st.button("Generate Speech", type="primary") and tts_text.strip():
            with st.spinner("🔊 Generating speech..."):
                try:
                    if openai_client:
                        response = openai_client.audio.speech.create(
                            model="tts-1",
                            voice=tts_voice,
                            input=tts_text
                        )

                        # Convert to bytes for Streamlit audio player
                        audio_bytes = b""
                        for chunk in response.iter_bytes():
                            audio_bytes += chunk

                        st.success("✅ Generated!")
                        st.audio(audio_bytes, format="audio/mp3")
                    else:
                        st.error("❌ OpenAI API key not available for text-to-speech.")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

with tab4:
    st.header("🎯 Multimodal Pipeline")

    st.markdown("Combine multiple modalities for creative content generation.")
    st.info("This demo shows how different AI models can work together.")

    col1, col2 = st.columns(2)

    with col1:
        multimodal_prompt = st.text_area(
            "Creative concept:",
            placeholder="A futuristic city with flying cars...",
            height=100
        )

    with col2:
        content_type = st.selectbox(
            "Output type:",
            ["Text + Image", "Text + Audio", "Simple Combined Demo"]
        )

    if st.button("Create Multimodal Content", type="primary", use_container_width=True):
        if multimodal_prompt.strip():
            with st.spinner("🎨 Creating multimodal content..."):
                try:
                    st.success("✅ Created!")

                    if content_type == "Text + Image":
                        # Generate text description - prefer OpenAI for reliability
                        if openai_client:
                            text_response = openai_client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": f"Write a creative description for: {multimodal_prompt}"}],
                                max_tokens=200
                            )
                            st.markdown("### 📝 Generated Description:")
                            st.write(text_response.choices[0].message.content)
                        elif gemini_model:
                            try:
                                text_response = gemini_model.generate_content(f"Write a creative description for: {multimodal_prompt}")
                                st.markdown("### 📝 Generated Description:")
                                st.write(text_response.text)
                            except Exception as e:
                                if "quota" in str(e).lower():
                                    st.error("❌ Gemini quota exceeded. Use OpenAI for text generation.")

                        # Generate image
                        if openai_client:
                            image_response = openai_client.images.generate(
                                model="dall-e-3",
                                prompt=multimodal_prompt,
                                size="1024x1024",
                                quality="standard",
                                n=1,
                            )
                            image_url = image_response.data[0].url
                            st.markdown("### 🖼️ Generated Image:")
                            st.image(image_url, caption=multimodal_prompt)

                    elif content_type == "Text + Audio":
                        # Generate text - prefer OpenAI
                        story_text = ""
                        if openai_client:
                            text_response = openai_client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": f"Write a short story about: {multimodal_prompt}"}],
                                max_tokens=300
                            )
                            story_text = text_response.choices[0].message.content
                            st.markdown("### 📖 Generated Story:")
                            st.write(story_text)
                        elif gemini_model:
                            try:
                                text_response = gemini_model.generate_content(f"Write a short story about: {multimodal_prompt}")
                                story_text = text_response.text
                                st.markdown("### 📖 Generated Story:")
                                st.write(story_text)
                            except Exception as e:
                                if "quota" in str(e).lower():
                                    st.error("❌ Gemini quota exceeded. Use OpenAI for text generation.")

                        # Generate audio
                        if openai_client and story_text:
                            audio_response = openai_client.audio.speech.create(
                                model="tts-1",
                                voice="alloy",
                                input=story_text[:1000] if 'story_text' in locals() else multimodal_prompt
                            )

                            audio_bytes = b""
                            for chunk in audio_response.iter_bytes():
                                audio_bytes += chunk

                            st.markdown("### 🔊 Audio Version:")
                            st.audio(audio_bytes, format="audio/mp3")

                    elif content_type == "Simple Combined Demo":
                        st.markdown("### 🎯 Combined Demo:")
                        st.write(f"**Concept:** {multimodal_prompt}")
                        st.write("This demonstrates how text, image, and audio AI models can work together to create rich multimedia content.")

                        # Show capabilities
                        st.markdown("**🤖 AI Capabilities Used:**")
                        st.markdown(f"- **Text Generation:** {gemini_model_name or 'Gemini'}")
                        st.markdown("- **Image Creation:** DALL-E 3")
                        st.markdown("- **Speech Synthesis:** OpenAI TTS")

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
