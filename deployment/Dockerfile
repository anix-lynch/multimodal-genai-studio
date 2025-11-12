# Multimodal GenAI Studio - Docker Configuration

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p outputs uploads cache logs

# Expose Gradio port
EXPOSE 7861

# Run the app
CMD ["python", "app.py"]

