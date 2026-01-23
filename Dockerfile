# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_ALLOC_CONF=expandable_segments:True
ENV MODEL_NAME=crestai/spark-tts-nexvox-v2
ENV TOKENIZER_REPO=unsloth/Spark-TTS-0.5B
ENV TOKENIZER_CACHE_DIR=Spark-TTS-0.5B
ENV SPARK_TTS_REPO_PATH=Spark-TTS
ENV HOST=0.0.0.0
ENV PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clone Spark-TTS repository
RUN git clone https://github.com/SparkAudio/Spark-TTS.git

# Copy application code
COPY spark_tts_streaming.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["python", "spark_tts_streaming.py"]
