#!/bin/bash
# Multimodal GenAI Studio - Setup Script

echo "🎨 Multimodal GenAI Studio - Setup"
echo "===================================="

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python $python_version found"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
echo "✅ Virtual environment created"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✅ Pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file..."
    cat > .env << 'EOF'
# Multimodal GenAI Studio - Environment Variables
# Edit this file with your actual API keys

# === Minimal Setup (FREE) ===
# Get Gemini API key: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=

# === Optional: For Full Features ===
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
HF_TOKEN=

# === Server Settings ===
HOST=0.0.0.0
PORT=7861
EOF
    echo "✅ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file and add your API key(s)"
    echo "   Minimum: GOOGLE_API_KEY (FREE from https://makersuite.google.com/app/apikey)"
else
    echo ""
    echo "✅ .env file already exists"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p outputs/images outputs/audio outputs/transcriptions outputs/multimodal
mkdir -p uploads cache logs data/examples
echo "✅ Directories created"

# Check configuration
echo ""
echo "Checking configuration..."
python3 config.py

echo ""
echo "===================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API key(s)"
echo "2. Run: python app.py"
echo "3. Open: http://localhost:7861"
echo ""
echo "Quick start: See QUICKSTART.md"
echo "===================================="

