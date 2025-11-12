# Deploy to HuggingFace Spaces

## Quick Deploy

### Option 1: Web UI (Recommended)

1. **Create Space**
   - Go to https://huggingface.co/new-space
   - Name: `multimodal-genai-studio`
   - SDK: `Gradio`
   - Hardware: `CPU basic` (FREE)
   - Visibility: `Public`

2. **Upload Files**
   - Upload all files from project root:
     - `app.py`
     - `config.py`
     - `requirements.txt`
     - `src/` folder (entire directory)
     - `.gitignore`

3. **Configure Secrets**
   Go to Settings → Repository secrets → Add:
   
   ```
   GOOGLE_API_KEY=your_google_api_key
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   HF_TOKEN=your_huggingface_token
   ```
   
   **Note:** At minimum, configure ONE API key to enable features.

4. **Wait for Build**
   - Space will build automatically (~3-5 minutes)
   - Check build logs for any errors
   - App will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/multimodal-genai-studio`

### Option 2: Git CLI

```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/multimodal-genai-studio
cd multimodal-genai-studio

# Copy project files
cp -r /path/to/multimodal-genai-studio/* .

# Push to HF
git add .
git commit -m "Initial deploy"
git push
```

## Configuration

### Minimal Setup (FREE)
If you only want to use free services:

```bash
GOOGLE_API_KEY=your_gemini_key  # FREE: 60 req/min
```

This enables:
- ✅ Text generation (Gemini)
- ✅ Text-to-Speech (gTTS - built-in)
- ✅ Multimodal pipelines

### Full Setup
For all features:

```bash
GOOGLE_API_KEY=your_gemini_key      # Text generation
OPENAI_API_KEY=your_openai_key      # Image gen, better audio
ANTHROPIC_API_KEY=your_claude_key   # Alternative text model
HF_TOKEN=your_hf_token              # Stable Diffusion
```

## Hardware Options

### CPU Basic (FREE)
- Best for: Text generation, light usage
- Supports: All features (slower image generation)
- Cost: $0

### CPU Upgrade ($0.03/hour)
- Best for: Regular usage
- Supports: All features (faster)
- Cost: ~$22/month (continuous)

### T4 Small GPU ($0.60/hour)
- Best for: Heavy image generation
- Supports: All features (very fast)
- Cost: ~$432/month (continuous)

**Recommendation:** Start with CPU Basic (FREE)

## Custom Domain

After deployment, you can:

1. **HuggingFace subdomain:**
   - `https://YOUR_USERNAME-multimodal-genai-studio.hf.space`

2. **Embed in your site:**
   ```html
   <iframe
     src="https://YOUR_USERNAME-multimodal-genai-studio.hf.space"
     width="100%"
     height="1000px"
   ></iframe>
   ```

3. **Custom domain:**
   - Upgrade to HF Pro
   - Configure custom domain in Space settings

## Troubleshooting

### Build Fails
- Check `requirements.txt` for incompatible versions
- Verify Python version (3.11)
- Check build logs for specific errors

### App Doesn't Start
- Verify at least one API key is set
- Check secrets are named correctly
- Review app logs in Space

### Features Not Working
- Verify API keys in secrets
- Check API key has correct permissions
- Review model availability in logs

## Monitoring

- **Logs:** Check Space logs for errors
- **Usage:** Monitor API usage in respective platforms
- **Performance:** Check Space analytics

## Updates

To update your deployment:

```bash
# Make changes locally
git add .
git commit -m "Update: description"
git push
```

Space will rebuild automatically.

## Cost Optimization

### FREE Tier Strategy
1. Use Gemini for text (FREE)
2. Use gTTS for speech (FREE)
3. Use HuggingFace API for images (FREE tier)
4. Deploy on CPU Basic (FREE)

**Total cost: $0/month** ✅

### Paid Tier Strategy
For production with high traffic:
1. OpenAI for better quality
2. CPU Upgrade or GPU for speed
3. Consider rate limits

**Estimate: $25-50/month**

## Support

- HuggingFace Docs: https://huggingface.co/docs/hub/spaces
- Gradio Docs: https://gradio.app/docs
- Project Issues: [Create issue in repo]

