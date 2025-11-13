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

def get_openrouter_client():
    """Get OpenRouter API client (alternative to OpenAI)"""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            return OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        return None
    except ImportError:
        return None

def get_anthropic_client():
    """Get Anthropic Claude client"""
    try:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            return anthropic.Anthropic(api_key=api_key)
        return None
    except ImportError:
        return None

# Initialize clients
gemini_model, gemini_model_name = get_gemini_client()
openai_client = get_openai_client()
openrouter_client = get_openrouter_client()
anthropic_client = get_anthropic_client()

# Header
st.title("🎨 Multimodal GenAI Studio")
st.markdown("Create and combine text, images, and audio using various AI models.")
st.markdown("**🎓 IBM Coursera Certification:** Build Multimodal Generative AI Applications")

# Sidebar for API status
with st.sidebar:
    st.header("🔑 API Status")

    # Check API availability
    gemini_available = gemini_model is not None
    openai_available = openai_client is not None
    openrouter_available = openrouter_client is not None
    anthropic_available = anthropic_client is not None

    # Show status with recommendations
    if gemini_available and gemini_model_name:
        st.write(f"**Gemini ({gemini_model_name}):** ⚠️ (Free tier - quotas)")
    else:
        st.write("**Gemini:** ❌")

    st.write(f"**OpenAI:** {'❌ Quota exceeded' if not openai_available else '✅'}")
    st.write(f"**OpenRouter:** {'✅ Recommended' if openrouter_available else '❌'}")
    st.write(f"**Claude (Anthropic):** {'✅ Alternative' if anthropic_available else '❌'}")

    # Debug info
    if st.checkbox("Show debug info"):
        st.write("**Debug Info:**")
        keys = {
            "Gemini": os.getenv("GEMINI_API_KEY", ""),
            "OpenAI": os.getenv("OPENAI_API_KEY", ""),
            "OpenRouter": os.getenv("OPENROUTER_API_KEY", ""),
            "Anthropic": os.getenv("ANTHROPIC_API_KEY", "")
        }
        for name, key in keys.items():
            st.write(f"{name} key: {'✅' if key else '❌'}")

        if gemini_model_name:
            st.write(f"Gemini model: {gemini_model_name}")

    # Show working alternatives
    working_apis = []
    if openrouter_available: working_apis.append("OpenRouter")
    if anthropic_available: working_apis.append("Claude")
    if openai_available: working_apis.append("OpenAI")

    if working_apis:
        st.success(f"✅ Working APIs: {', '.join(working_apis)}")
    else:
        st.error("❌ No working APIs - check your keys!")

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

        # Build model options dynamically - prioritize working APIs
        model_options = []

        # Add working APIs in priority order
        if openrouter_available:
            model_options.extend(["openrouter/gpt-4o-mini", "openrouter/claude-3-haiku", "openrouter/gemini-pro"])
        if anthropic_available:
            model_options.append("claude-3-haiku")
        if openai_available:
            model_options.append("gpt-4o-mini")
        if gemini_available and gemini_model_name:
            model_options.append(gemini_model_name)

        # Ensure we have at least one option
        if not model_options:
            model_options = ["No models available"]

        model = st.selectbox(
            "Model:",
            model_options,
            index=0,
            help="Multiple APIs available - OpenRouter recommended for reliability"
        )

        max_tokens = st.slider("Max tokens:", 100, 2000, 500)

    with col2:
        temperature = st.slider("Creativity:", 0.0, 2.0, 0.7)
        system_prompt = st.text_area("System prompt (optional):", height=100)

    if st.button("Generate Text", type="primary", use_container_width=True):
        if prompt.strip():
            with st.spinner("🤖 Generating text..."):
                try:
                    result_text = None

                    # Handle different API providers
                    if model.startswith("openrouter/"):
                        # OpenRouter API - use correct model names
                        if openrouter_client:
                            actual_model = model.replace("openrouter/", "")
                            # Map to correct OpenRouter model names
                            model_mapping = {
                                "gpt-4o-mini": "openai/gpt-4o-mini",
                                "claude-3-haiku": "anthropic/claude-3-haiku",
                                "gemini-pro": "google/gemini-pro"
                            }
                            or_model = model_mapping.get(actual_model, actual_model)

                            messages = []
                            if system_prompt:
                                messages.append({"role": "system", "content": system_prompt})
                            messages.append({"role": "user", "content": prompt})

                            response = openrouter_client.chat.completions.create(
                                model=or_model,
                                messages=messages,
                                max_tokens=max_tokens,
                                temperature=temperature
                            )
                            result_text = response.choices[0].message.content
                        else:
                            st.error("❌ OpenRouter client not available")

                    elif model == "claude-3-haiku" and anthropic_client:
                        # Anthropic Claude
                        system_msg = system_prompt if system_prompt else ""
                        response = anthropic_client.messages.create(
                            model="claude-3-haiku-20240307",
                            max_tokens=max_tokens,
                            temperature=temperature,
                            system=system_msg,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        result_text = response.content[0].text

                    elif model == "gpt-4o-mini" and openai_client:
                        # OpenAI (if quota allows)
                        try:
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
                        except Exception as openai_error:
                            if "insufficient_quota" in str(openai_error) or "429" in str(openai_error):
                                st.error("❌ OpenAI Quota Exceeded")
                                st.info("💡 **Try OpenRouter or Claude instead!**")
                            else:
                                st.error(f"❌ OpenAI Error: {str(openai_error)}")
                            result_text = None

                    elif model == gemini_model_name and gemini_model:
                        # Gemini (last resort)
                        try:
                            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                            response = gemini_model.generate_content(full_prompt)
                            result_text = response.text
                        except Exception as gemini_error:
                            error_msg = str(gemini_error).lower()
                            if "quota" in error_msg or "429" in error_msg:
                                st.error("❌ Gemini Free Tier Quota Exceeded")
                                st.info("💡 **Try OpenRouter or Claude instead!**")
                            else:
                                st.error(f"❌ Gemini API Error: {str(gemini_error)}")
                            result_text = None

                    else:
                        st.error(f"❌ {model} model not available")
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
                    success = False

                    # Try Stability AI via OpenRouter (they have good free tier)
                    if openrouter_client and not success:
                        try:
                            response = openrouter_client.images.generate(
                                model="stability-ai/sdxl",
                                prompt=image_prompt,
                                size="1024x1024",
                                n=1,
                            )
                            if hasattr(response, 'data') and response.data:
                                image_url = response.data[0].url
                                st.success("✅ Generated with Stability AI!")
                                st.image(image_url, caption=image_prompt, use_container_width=True)
                                success = True
                        except Exception as e:
                            st.warning(f"Stability AI failed: {str(e)[:50]}...")

                    # Try another Stability AI model
                    if openrouter_client and not success:
                        try:
                            response = openrouter_client.images.generate(
                                model="stability-ai/sd3-medium",
                                prompt=image_prompt,
                                size="1024x1024",
                                n=1,
                            )
                            if hasattr(response, 'data') and response.data:
                                image_url = response.data[0].url
                                st.success("✅ Generated with SD3!")
                                st.image(image_url, caption=image_prompt, use_container_width=True)
                                success = True
                        except Exception as e:
                            st.warning(f"SD3 failed: {str(e)[:50]}...")

                    # Try FLUX model
                    if openrouter_client and not success:
                        try:
                            response = openrouter_client.images.generate(
                                model="blackforestlabs/flux-1.1-pro",
                                prompt=image_prompt,
                                size="1024x1024",
                                n=1,
                            )
                            if hasattr(response, 'data') and response.data:
                                image_url = response.data[0].url
                                st.success("✅ Generated with FLUX!")
                                st.image(image_url, caption=image_prompt, use_container_width=True)
                                success = True
                        except Exception as e:
                            st.warning(f"FLUX failed: {str(e)[:50]}...")

                    # Try OpenAI as fallback (if quota allows)
                    if openai_client and not success:
                        try:
                            response = openai_client.images.generate(
                                model=image_model,
                                prompt=image_prompt,
                                size=image_size,
                                quality="standard",
                                n=1,
                            )
                            image_url = response.data[0].url
                            st.success("✅ Generated with OpenAI!")
                            st.image(image_url, caption=image_prompt, use_container_width=True)
                            success = True
                        except Exception as e:
                            if "billing_hard_limit" in str(e) or "insufficient_quota" in str(e):
                                st.warning("OpenAI billing limit reached - try OpenRouter above")
                            else:
                                st.error(f"OpenAI failed: {str(e)[:50]}...")

                    if not success:
                        st.error("❌ All image generation services failed. Try different models or check API keys.")

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
                    success = False

                    # Try OpenRouter first (various Whisper models)
                    if openrouter_client and not success:
                        try:
                            audio_bytes = audio_file.getvalue()
                            transcription = openrouter_client.audio.transcriptions.create(
                                model="openai/whisper-large-v3",
                                file=("audio." + audio_file.type.split('/')[-1], audio_bytes, audio_file.type)
                            )
                            st.success("✅ Transcribed with OpenRouter!")
                            st.markdown("### Transcription:")
                            st.write(transcription.text)
                            success = True
                        except Exception as e:
                            st.warning(f"OpenRouter transcription failed: {str(e)[:50]}...")

                    # Try OpenAI as fallback
                    if openai_client and not success:
                        try:
                            audio_bytes = audio_file.getvalue()
                            transcription = openai_client.audio.transcriptions.create(
                                model="whisper-1",
                                file=("audio." + audio_file.type.split('/')[-1], audio_bytes, audio_file.type)
                            )
                            st.success("✅ Transcribed with OpenAI!")
                            st.markdown("### Transcription:")
                            st.write(transcription.text)
                            success = True
                        except Exception as e:
                            if "billing_hard_limit" in str(e) or "insufficient_quota" in str(e):
                                st.warning("OpenAI billing limit reached - try OpenRouter above")
                            else:
                                st.error(f"OpenAI transcription failed: {str(e)[:50]}...")

                    if not success:
                        st.error("❌ All transcription services failed.")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    with audio_tab2:
        st.subheader("Text to Speech (OpenAI TTS)")
        tts_text = st.text_area("Enter text to convert to speech:", height=100, placeholder="Hello, this is a sample text to speech conversion.")
        tts_voice = st.selectbox("Voice:", ["alloy", "echo", "fable", "onyx"], index=0)

        if st.button("Generate Speech", type="primary") and tts_text.strip():
            with st.spinner("🔊 Generating speech..."):
                try:
                    success = False

                    # Try OpenRouter first (various TTS models)
                    if openrouter_client and not success:
                        try:
                            response = openrouter_client.audio.speech.create(
                                model="openai/tts-1",
                                voice=tts_voice if tts_voice in ["alloy", "echo", "fable", "onyx"] else "alloy",
                                input=tts_text[:4000]  # Limit text length
                            )

                            audio_bytes = b""
                            for chunk in response.iter_bytes():
                                audio_bytes += chunk

                            st.success("✅ Generated with OpenRouter!")
                            st.audio(audio_bytes, format="audio/mp3")
                            success = True
                        except Exception as e:
                            st.warning(f"OpenRouter TTS failed: {str(e)[:50]}...")

                    # Try OpenAI as fallback
                    if openai_client and not success:
                        try:
                            response = openai_client.audio.speech.create(
                                model="tts-1",
                                voice=tts_voice,
                                input=tts_text
                            )

                            audio_bytes = b""
                            for chunk in response.iter_bytes():
                                audio_bytes += chunk

                            st.success("✅ Generated with OpenAI!")
                            st.audio(audio_bytes, format="audio/mp3")
                            success = True
                        except Exception as e:
                            if "billing_hard_limit" in str(e) or "insufficient_quota" in str(e):
                                st.warning("OpenAI billing limit reached - try OpenRouter above")
                            else:
                                st.error(f"OpenAI TTS failed: {str(e)[:50]}...")

                    if not success:
                        st.error("❌ All text-to-speech services failed.")

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
                        # Generate text description - try multiple APIs
                        text_generated = False
                        description_text = ""

                        if openrouter_client and not text_generated:
                            try:
                                text_response = openrouter_client.chat.completions.create(
                                    model="openai/gpt-4o-mini",
                                    messages=[{"role": "user", "content": f"Write a creative description for: {multimodal_prompt}"}],
                                    max_tokens=200
                                )
                                description_text = text_response.choices[0].message.content
                                st.markdown("### 📝 Generated Description (OpenRouter):")
                                st.write(description_text)
                                text_generated = True
                            except Exception as e:
                                st.warning(f"OpenRouter text failed: {str(e)[:50]}...")

                        if openai_client and not text_generated:
                            try:
                                text_response = openai_client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[{"role": "user", "content": f"Write a creative description for: {multimodal_prompt}"}],
                                    max_tokens=200
                                )
                                description_text = text_response.choices[0].message.content
                                st.markdown("### 📝 Generated Description (OpenAI):")
                                st.write(description_text)
                                text_generated = True
                            except Exception as e:
                                if "billing_hard_limit" in str(e):
                                    st.warning("OpenAI billing limit reached for text")

                        if anthropic_client and not text_generated:
                            try:
                                response = anthropic_client.messages.create(
                                    model="claude-3-haiku-20240307",
                                    max_tokens=200,
                                    messages=[{"role": "user", "content": f"Write a creative description for: {multimodal_prompt}"}]
                                )
                                description_text = response.content[0].text
                                st.markdown("### 📝 Generated Description (Claude):")
                                st.write(description_text)
                                text_generated = True
                            except Exception as e:
                                st.warning(f"Claude text failed: {str(e)[:50]}...")

                        # Generate real image - try Stability AI first
                        if openrouter_client:
                            try:
                                response = openrouter_client.images.generate(
                                    model="stability-ai/sdxl",
                                    prompt=f"{multimodal_prompt}, highly detailed, professional image",
                                    size="1024x1024",
                                    n=1,
                                )
                                if hasattr(response, 'data') and response.data:
                                    image_url = response.data[0].url
                                    st.markdown("### 🖼️ Generated Image (Stability AI):")
                                    st.image(image_url, caption=multimodal_prompt, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Stability AI image failed: {str(e)[:50]}...")
                                # Try FLUX as fallback
                                try:
                                    response = openrouter_client.images.generate(
                                        model="blackforestlabs/flux-1.1-pro",
                                        prompt=f"{multimodal_prompt}, highly detailed, professional image",
                                        size="1024x1024",
                                        n=1,
                                    )
                                    if hasattr(response, 'data') and response.data:
                                        image_url = response.data[0].url
                                        st.markdown("### 🖼️ Generated Image (FLUX):")
                                        st.image(image_url, caption=multimodal_prompt, use_container_width=True)
                                except Exception as e2:
                                    st.warning(f"FLUX image failed: {str(e)[:50]}...")

                    elif content_type == "Text + Audio":
                        # Generate text - try multiple APIs
                        story_text = ""
                        text_generated = False

                        if openrouter_client and not text_generated:
                            try:
                                text_response = openrouter_client.chat.completions.create(
                                    model="openai/gpt-4o-mini",
                                    messages=[{"role": "user", "content": f"Write a short story about: {multimodal_prompt}"}],
                                    max_tokens=300
                                )
                                story_text = text_response.choices[0].message.content
                                st.markdown("### 📖 Generated Story (OpenRouter):")
                                st.write(story_text)
                                text_generated = True
                            except Exception as e:
                                st.warning(f"OpenRouter story failed: {str(e)[:50]}...")

                        if openai_client and not text_generated:
                            try:
                                text_response = openai_client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[{"role": "user", "content": f"Write a short story about: {multimodal_prompt}"}],
                                    max_tokens=300
                                )
                                story_text = text_response.choices[0].message.content
                                st.markdown("### 📖 Generated Story (OpenAI):")
                                st.write(story_text)
                                text_generated = True
                            except Exception as e:
                                if "billing_hard_limit" in str(e):
                                    st.warning("OpenAI billing limit reached for text")

                        # Generate audio - try OpenRouter first
                        if story_text:
                            if openrouter_client:
                                try:
                                    audio_response = openrouter_client.audio.speech.create(
                                        model="openai/tts-1",
                                        voice="alloy",
                                        input=story_text[:1000]
                                    )
                                    audio_bytes = b""
                                    for chunk in audio_response.iter_bytes():
                                        audio_bytes += chunk
                                    st.markdown("### 🔊 Audio Version (OpenRouter):")
                                    st.audio(audio_bytes, format="audio/mp3")
                                except Exception as e:
                                    st.warning(f"OpenRouter audio failed: {str(e)[:50]}...")
                                    # Try OpenAI as fallback
                                    if openai_client:
                                        try:
                                            audio_response = openai_client.audio.speech.create(
                                                model="tts-1",
                                                voice="alloy",
                                                input=story_text[:1000]
                                            )
                                            audio_bytes = b""
                                            for chunk in audio_response.iter_bytes():
                                                audio_bytes += chunk
                                            st.markdown("### 🔊 Audio Version (OpenAI):")
                                            st.audio(audio_bytes, format="audio/mp3")
                                        except Exception as e2:
                                            if "billing_hard_limit" in str(e2):
                                                st.warning("OpenAI billing limit reached for audio")

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
