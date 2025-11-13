---
title: Multimodal GenAI Studio
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# 🎨 Multimodal GenAI Studio

![Multimodal GenAI Studio Demo](https://raw.githubusercontent.com/anix-lynch/multimodal-genai-studio/main/public/multimodal.gif)

![Thumbnail](multimodal-thumbnail.png)

Professional multimodal AI application supporting text, image, and audio generation.

## Features

- **Text Generation**: Google Gemini, OpenAI GPT, Anthropic Claude
- **Image Generation**: DALL-E 2/3, Stable Diffusion
- **Audio**: Whisper transcription, TTS (OpenAI + gTTS)
- **Multimodal Pipelines**: 5 combined workflows

## FREE Setup

Works with just Gemini API key (FREE tier):
- Text generation (Gemini)
- Text-to-speech (gTTS)
- Multimodal pipelines

## Configuration

Add secrets in Space settings:
- `GOOGLE_API_KEY` - Gemini (required)
- `OPENAI_API_KEY` - GPT + DALL-E + Whisper (optional)
- `ANTHROPIC_API_KEY` - Claude (optional)
- `HF_TOKEN` - Stable Diffusion (optional)
