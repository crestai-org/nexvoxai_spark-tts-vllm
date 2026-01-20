# Spark TTS Streaming Server

A high-performance FastAPI server for streaming text-to-speech using the Spark TTS model with vLLM. Supports WebSocket and HTTP streaming with multiple African language voices.

## Features

- **6 Voice Options**: Acholi, Ateso, Runyankore, Lugbara, Swahili, and Luganda
- **Fast Streaming**: Sentence-by-sentence generation for low latency
- **Dual Protocols**: WebSocket and HTTP streaming endpoints
- **vLLM Backend**: Efficient inference with GPU acceleration
- **Multi-language Support**: African languages with native speakers
- **Easy Integration**: REST API compatible with standard tools

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (tested on RTX 4090, 24GB VRAM)
- Hugging Face account with access token

### Installation

1. **Clone the repository and dependencies:**

```bash
# Clone Spark-TTS repository
git clone https://github.com/SparkAudio/Spark-TTS
cd Spark-TTS

# Install dependencies
pip install einx einops soundfile librosa vllm omegaconf fastapi uvicorn websockets
```

2. **Set up Hugging Face authentication:**

```bash
# Login to Hugging Face
huggingface-cli login
# Or set token directly
export HF_TOKEN=your_token_here
```

3. **Download the server script:**

Save the `spark_tts_streaming_server.py` file to your project directory.

### Running the Server

```bash
python spark_tts_streaming_server.py
```

The server will start on `http://0.0.0.0:8001`

## API Reference

### Available Voices

| Voice Name | Speaker ID | Language | Gender |
|------------|------------|----------|--------|
| `acholi_female` | 241 | Acholi | Female |
| `ateso_female` | 242 | Ateso | Female |
| `runyankore_female` | 243 | Runyankore | Female |
| `lugbara_female` | 245 | Lugbara | Female |
| `swahili_male` | 246 | Swahili | Male |
| `luganda_female` | 248 | Luganda | Female |

### Using Docker Compose (Recommended)

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

### Endpoints

#### 1. WebSocket Streaming

**Endpoint:** `ws://localhost:8001/v1/audio/speech/stream/ws`

**Protocol:**

**Client → Server:**
```json
{
  "input": "Your text here",
  "voice": "luganda_female",
  "speaker_id": 248,
  "temperature": 0.7,
  "segment_id": "unique_id",
  "continue": true
}
```

**Server → Client:**
```json
// Start message
{"type": "start", "segment_id": "unique_id", "speaker_id": 248}

// Binary audio chunks (PCM16, 16kHz, mono)
<binary data>

// End message
{"type": "end", "segment_id": "unique_id"}

// Error message (if any)
{"type": "error", "message": "error description"}
```

#### 2. HTTP Streaming

**Endpoint:** `POST /v1/audio/speech/stream`

**Request Body:**
```json
{
  "text": "Your text here",
  "voice": "luganda_female",
  "speaker_id": 248,
  "temperature": 0.7,
  "max_tokens": 2048
}
```

**Response:**
- Content-Type: `audio/pcm`
- Headers:
  - `X-Sample-Rate: 16000`
  - `X-Bit-Depth: 16`
  - `X-Channels: 1`
- Body: Streaming PCM audio data

#### 3. List Available Voices

**Endpoint:** `GET /v1/voices`

**Response:**
```json
{
  "voices": [
    {
      "id": "luganda_female",
      "speaker_id": 248,
      "language": "luganda"
    }
  ]
}
```

#### 4. Server Info

**Endpoint:** `GET /`

Returns server information, available voices, and usage examples.

## Usage Examples

### Python - WebSocket Client

```python
import asyncio
import websockets
import json
import wave

async def generate_speech():
    uri = "ws://localhost:8001/v1/audio/speech/stream/ws"
    
    async with websockets.connect(uri) as ws:
        # Send text for generation
        await ws.send(json.dumps({
            "input": "Nze Prosi Nafula. Ndi musawo akola ku bantu abalina kookolo.",
            "voice": "luganda_female",
            "temperature": 0.7,
            "segment_id": "test_1"
        }))
        
        audio_data = bytearray()
        
        while True:
            message = await ws.recv()
            
            if isinstance(message, bytes):
                # Audio chunk received
                audio_data.extend(message)
                print(f"Received {len(message)} bytes")
            else:
                # Control message
                data = json.loads(message)
                print(f"Control: {data}")
                
                if data.get("type") == "end":
                    break
                elif data.get("type") == "error":
                    print(f"Error: {data.get('message')}")
                    break
        
        # Save to WAV file
        with wave.open("output.wav", "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz
            wav_file.writeframes(bytes(audio_data))
        
        print("Audio saved to output.wav")

asyncio.run(generate_speech())
```

### Python - HTTP Client

```python
import requests
import wave

def generate_speech_http():
    url = "http://localhost:8001/v1/audio/speech/stream"
    
    response = requests.post(
        url,
        json={
            "text": "Habari, naitwa Prosi Nafula. Mimi ni muuguzi.",
            "voice": "swahili_male",
            "temperature": 0.7
        },
        stream=True
    )
    
    audio_data = bytearray()
    
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            audio_data.extend(chunk)
            print(f"Received {len(chunk)} bytes")
    
    # Save to WAV file
    with wave.open("output_http.wav", "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(bytes(audio_data))
    
    print("Audio saved to output_http.wav")

generate_speech_http()
```

### JavaScript - WebSocket Client

```javascript
const ws = new WebSocket('ws://localhost:8001/v1/audio/speech/stream/ws');

const audioChunks = [];

ws.onopen = () => {
    ws.send(JSON.stringify({
        input: "Nze Prosi Nafula. Ndi musawo.",
        voice: "luganda_female",
        temperature: 0.7,
        segment_id: "js_test_1"
    }));
};

ws.onmessage = (event) => {
    if (event.data instanceof Blob) {
        // Audio chunk
        audioChunks.push(event.data);
        console.log(`Received audio chunk: ${event.data.size} bytes`);
    } else {
        // Control message
        const data = JSON.parse(event.data);
        console.log('Control message:', data);
        
        if (data.type === 'end') {
            // Combine chunks and play
            const audioBlob = new Blob(audioChunks, { type: 'audio/pcm' });
            // Process audioBlob...
            ws.close();
        }
    }
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};
```

### cURL - HTTP Endpoint

```bash
curl -X POST http://localhost:8001/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Nze Prosi Nafula. Ndi musawo.",
    "voice": "luganda_female",
    "temperature": 0.7
  }' \
  --output output.pcm

# Convert PCM to WAV
ffmpeg -f s16le -ar 16000 -ac 1 -i output.pcm output.wav
```

## Configuration

### Environment Variables

```bash
# GPU selection
export CUDA_VISIBLE_DEVICES=0

# Model configuration
export MODEL_NAME=crestai/spark-tts-nexvox
export TOKENIZER_REPO=unsloth/Spark-TTS-0.5B
export TOKENIZER_CACHE_DIR=Spark-TTS-0.5B

# Spark-TTS repository path
export SPARK_TTS_REPO_PATH=Spark-TTS

# Server configuration
export HOST=0.0.0.0
export PORT=8001
```

### Server Parameters

Edit in `spark_tts_streaming_server.py`:

```python
# Default parameters
DEFAULT_TEMPERATURE = 0.7  # 0.1 (conservative) to 1.0 (creative)
DEFAULT_MAX_TOKENS = 2048  # Maximum tokens per generation
DEFAULT_SPEAKER_ID = 248   # Default voice

# Audio configuration
AUDIO_SAMPLERATE = 16000   # 16kHz sample rate
AUDIO_BITS_PER_SAMPLE = 16 # 16-bit audio
AUDIO_CHANNELS = 1         # Mono audio
```

## Performance Tips

### Memory Management

- **GPU Memory**: Server uses 50% GPU memory by default (`gpu_memory_utilization=0.5`)
- **Concurrent Requests**: WebSocket allows multiple concurrent connections
- **Sentence Chunking**: Automatic text splitting for efficient streaming

### Optimization

1. **Temperature Settings:**
   - `0.1-0.3`: More consistent, less varied
   - `0.5-0.7`: Balanced (recommended)
   - `0.8-1.0`: More creative, potentially less stable

2. **Text Length:**
   - Short texts (1-3 sentences): ~1-2 seconds
   - Medium texts (5-10 sentences): ~3-5 seconds
   - Long texts: Automatically chunked

3. **Hardware Requirements:**
   - Minimum: 16GB VRAM
   - Recommended: 24GB VRAM (RTX 4090, A5000)
   - CPU: Multi-core for async processing

## Audio Format

**Output Specification:**
- Format: PCM (uncompressed)
- Sample Rate: 16,000 Hz
- Bit Depth: 16-bit
- Channels: Mono (1)
- Byte Order: Little-endian
- Encoding: Signed integer

**Converting to Common Formats:**

```bash
# PCM to WAV
ffmpeg -f s16le -ar 16000 -ac 1 -i input.pcm output.wav

# PCM to MP3
ffmpeg -f s16le -ar 16000 -ac 1 -i input.pcm -b:a 128k output.mp3

# PCM to OGG
ffmpeg -f s16le -ar 16000 -ac 1 -i input.pcm -c:a libvorbis output.ogg
```

## Troubleshooting

### Common Issues

**1. "Could not import BiCodecTokenizer"**
```bash
# Make sure Spark-TTS is cloned
git clone https://github.com/SparkAudio/Spark-TTS
export SPARK_TTS_REPO_PATH=./Spark-TTS
```

**2. "CUDA out of memory"**
```python
# Reduce GPU memory utilization in code
vllm_model = LLM(
    MODEL_NAME,
    gpu_memory_utilization=0.3  # Reduce from 0.5
)
```

**3. "No semantic tokens found"**
- Increase `max_tokens` parameter
- Check if input text is valid
- Try different temperature values

**4. WebSocket connection refused**
```bash
# Check if server is running
curl http://localhost:8001/

# Check firewall settings
sudo ufw allow 8001
```

## Development

### Running in Development Mode

```bash
# With auto-reload (slower startup)
uvicorn spark_tts_streaming_server:app --reload --host 0.0.0.0 --port 8001

# Production mode
python spark_tts_streaming_server.py
```

### Testing

```python
# Test voices endpoint
curl http://localhost:8001/v1/voices

# Test health check
curl http://localhost:8001/

# Test streaming
python test_client.py
```

## License

This project uses:
- **Spark TTS**: [License](https://github.com/SparkAudio/Spark-TTS)
- **vLLM**: Apache 2.0 License
- **FastAPI**: MIT License

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions:
- GitHub Issues: [Create an issue]
- Spark TTS: https://github.com/SparkAudio/Spark-TTS
- vLLM Docs: https://docs.vllm.ai/

## Acknowledgments

- Spark Audio team for the Spark TTS model
- vLLM team for efficient LLM inference
- Anthropic for Claude assistance

---

