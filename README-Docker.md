# Docker Deployment Guide

## Prerequisites

1. **Docker Engine** (20.10+)
2. **Docker Compose** (2.0+)
3. **NVIDIA Container Toolkit** (for GPU support)
4. **NVIDIA GPU** with CUDA support

## Installation

### 1. Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 2. Verify GPU Support

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and start the service
docker compose up --build

# Run in detached mode
docker compose up -d --build

# View logs
docker compose logs -f

# Stop the service
docker compose down
```

### Option 2: Using Docker Directly

```bash
# Build the image
docker build -t spark-tts-streaming .

# Run the container
docker run --gpus all -p 8000:8000 --name spark-tts spark-tts-streaming

# Run in detached mode
docker run -d --gpus all -p 8000:8000 --name spark-tts spark-tts-streaming
```

## Configuration

### Environment Variables

You can customize the application using environment variables:

```bash
# GPU selection
CUDA_VISIBLE_DEVICES=0

# Model configuration
MODEL_NAME=crestai/spark-tts-nexvox-v2
TOKENIZER_REPO=unsloth/Spark-TTS-0.5B
TOKENIZER_CACHE_DIR=Spark-TTS-0.5B
SPARK_TTS_REPO_PATH=Spark-TTS
```

### Custom docker-compose.yml

Create a custom `docker-compose.override.yml` file:

```yaml
version: '3.8'

services:
  spark-tts:
    environment:
      - MODEL_NAME=your-custom-model
      - CUDA_VISIBLE_DEVICES=1
    ports:
      - "8080:8000"  # Use different host port
```

## API Endpoints

Once the container is running, the API will be available at:

- **HTTP API**: `http://localhost:8000`
- **WebSocket**: `ws://localhost:8000/v1/audio/speech/stream/ws`
- **Health Check**: `http://localhost:8000/`
- **Voices List**: `http://localhost:8000/v1/voices`

## Usage Examples

### HTTP Streaming

```bash
curl -X POST "http://localhost:8000/v1/audio/speech/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the Spark TTS streaming API.",
    "voice": "luganda_female",
    "temperature": 0.7
  }' \
  --output audio.pcm
```

### WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/v1/audio/speech/stream/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    input: "Hello, this is a WebSocket test.",
    voice: "luganda_female",
    segment_id: "test_1"
  }));
};

ws.onmessage = (event) => {
  if (event.data instanceof Blob) {
    // Handle audio data
    console.log('Received audio chunk');
  } else {
    // Handle JSON messages
    const message = JSON.parse(event.data);
    console.log('Received:', message);
  }
};
```

## Persistent Storage

The docker-compose.yml includes volume mounts for:

- `./cache:/app/cache` - Model cache
- `./Spark-TTS-0.5B:/app/Spark-TTS-0.5B` - Tokenizer cache

This ensures that downloaded models persist across container restarts.

## Monitoring

### Health Checks

The container includes built-in health checks:

```bash
# Check container health
docker ps

# View health check logs
docker inspect spark-tts-streaming | grep Health -A 10
```

### Logs

```bash
# View real-time logs
docker-compose logs -f spark-tts

# View last 100 lines
docker-compose logs --tail=100 spark-tts
```

## Troubleshooting

### Common Issues

1. **GPU not detected**: Ensure NVIDIA Container Toolkit is installed
2. **Out of memory**: Reduce `gpu_memory_utilization` in the code or use a smaller model
3. **Port conflicts**: Change the host port mapping in docker-compose.yml
4. **Permission issues**: Ensure proper Docker permissions

### Debug Mode

Run the container with additional debugging:

```bash
docker run --gpus all -p 8000:8000 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v $(pwd):/app \
  --entrypoint /bin/bash \
  spark-tts-streaming
```

## Performance Optimization

1. **GPU Memory**: Adjust `gpu_memory_utilization` based on available GPU memory
2. **Model Caching**: Use persistent volumes to avoid re-downloading models
3. **Concurrent Requests**: Scale horizontally using multiple containers
4. **Network**: Use host networking for better performance in production

## Production Deployment

For production use, consider:

1. **Load Balancing**: Use nginx or traefik for load balancing
2. **Monitoring**: Add Prometheus metrics and Grafana dashboards
3. **Security**: Implement authentication and rate limiting
4. **Resource Limits**: Set appropriate memory and CPU limits
5. **Backup**: Regularly backup model cache and configurations
