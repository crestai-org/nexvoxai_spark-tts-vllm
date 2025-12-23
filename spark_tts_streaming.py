import os
import sys
import torch
import numpy as np
import re
import json
import asyncio
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from vllm import LLM
from vllm.sampling_params import SamplingParams
from huggingface_hub import snapshot_download

# Configuration
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
MODEL_NAME = os.environ.get("MODEL_NAME", "crestai/spark-tts-nexvox")
TOKENIZER_REPO = os.environ.get("TOKENIZER_REPO", "unsloth/Spark-TTS-0.5B")
TOKENIZER_CACHE_DIR = os.environ.get("TOKENIZER_CACHE_DIR", "Spark-TTS-0.5B")
SPARK_TTS_REPO_PATH = os.environ.get("SPARK_TTS_REPO_PATH", "Spark-TTS")

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# Audio configuration
AUDIO_SAMPLERATE = 16000
AUDIO_BITS_PER_SAMPLE = 16
AUDIO_CHANNELS = 1

# Default parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_SPEAKER_ID = 248  # Luganda female

# Speaker IDs mapping
SPEAKER_IDS = {
    "acholi_female": 241,
    "ateso_female": 242,
    "runyankore_female": 243,
    "lugbara_female": 245,
    "swahili_male": 246,
    "luganda_female": 248,
}

# Precomputed global tokens for each speaker
GLOBAL_IDS_BY_SPEAKER = {
    241: [1755, 1265, 184, 3545, 2718, 2405, 3237, 1360, 3621, 1850, 37, 3382, 736,
          3380, 3131, 2036, 244, 2128, 254, 2550, 3181, 764, 1277, 502, 2941, 1993,
          3556, 1428, 3505, 3245, 3506, 1540],
    242: [1367, 1522, 308, 4061, 1449, 2468, 2193, 1349, 3458, 2339, 1651, 3174,
          501, 3364, 3194, 2041, 442, 1061, 502, 2234, 2397, 358, 3829, 2490, 2031,
          1002, 3548, 586, 3445, 1419, 4093, 2908],
    243: [2051, 242, 2684, 4062, 2654, 2252, 353, 3657, 2759, 3254, 1649, 3366,
          1017, 3600, 3131, 3813, 1535, 1595, 1059, 237, 2158, 1174, 4085, 2174,
          3791, 990, 3274, 2693, 3829, 2271, 2650, 1689],
    245: [2031, 2545, 116, 4060, 746, 1385, 3301, 1312, 3638, 1846, 85, 3190, 1016,
          3384, 3134, 954, 244, 1104, 235, 2549, 3357, 508, 1278, 1974, 2621, 1896,
          3812, 2185, 3061, 2941, 1187, 5],
    246: [1811, 1138, 2873, 3309, 2639, 723, 3363, 974, 1612, 2531, 1769, 3376,
          933, 3848, 3195, 2180, 2359, 1275, 3493, 3260, 2279, 3715, 3508, 2433,
          4082, 1087, 3545, 1449, 160, 3531, 2908, 2094],
    248: [2559, 1523, 440, 3789, 1438, 373, 2212, 1248, 3369, 1847, 36, 3126, 480,
          3380, 3133, 2041, 248, 2384, 730, 2554, 3182, 1785, 1277, 1013, 2425,
          1932, 3560, 1177, 2736, 2430, 2722, 261]
}

app = FastAPI()

# Global model variables
vllm_model = None
audio_tokenizer = None
device = None


class AudioRequest(BaseModel):
    text: str
    voice: str = "luganda_female"
    speaker_id: Optional[int] = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS


def chunk_text_simple(text: str) -> List[str]:
    """Split text into individual sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def initialize_models():
    """Initialize vLLM model and audio tokenizer."""
    global vllm_model, audio_tokenizer, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Add Spark-TTS to path if exists
    if os.path.exists(SPARK_TTS_REPO_PATH):
        sys.path.append(SPARK_TTS_REPO_PATH)
        print(f"Added {SPARK_TTS_REPO_PATH} to Python path")
    else:
        print(f"Warning: {SPARK_TTS_REPO_PATH} not found. Clone it with:")
        print(f"git clone https://github.com/SparkAudio/Spark-TTS")
    
    # Load vLLM model
    print(f"Loading Spark TTS model: {MODEL_NAME}...")
    vllm_model = LLM(
        MODEL_NAME,
        enforce_eager=False,
        gpu_memory_utilization=0.85
    )
    print("✅ Model loaded successfully!")
    
    # Download tokenizer if needed
    if not os.path.exists(TOKENIZER_CACHE_DIR):
        print(f"Downloading tokenizer from {TOKENIZER_REPO}...")
        snapshot_download(
            repo_id=TOKENIZER_REPO,
            local_dir=TOKENIZER_CACHE_DIR,
            ignore_patterns=["*LLM*"],
        )
        print(f"✅ Tokenizer downloaded to {TOKENIZER_CACHE_DIR}")
    
    # Initialize audio tokenizer
    try:
        from sparktts.models.audio_tokenizer import BiCodecTokenizer
        print("Initializing audio tokenizer...")
        audio_tokenizer = BiCodecTokenizer(TOKENIZER_CACHE_DIR, device)
        print("✅ Audio tokenizer initialized!")
    except ImportError:
        print("Error: Could not import BiCodecTokenizer. Make sure Spark-TTS repo is available.")
        raise


def generate_audio_segment(text: str, speaker_id: int, temperature: float) -> np.ndarray:
    """Generate audio for a single text segment."""
    global_tokens = GLOBAL_IDS_BY_SPEAKER[speaker_id]
    
    # Create prompt
    prompt = f"<|task_tts|><|start_content|>{speaker_id}: {text}<|end_content|><|start_global_token|>"
    prompt += ''.join([f'<|bicodec_global_{t}|>' for t in global_tokens])
    prompt += '<|end_global_token|><|start_semantic_token|>'
    
    # Generate with vLLM
    sampling_params = SamplingParams(temperature=temperature, max_tokens=DEFAULT_MAX_TOKENS)
    outputs = vllm_model.generate(prompts=[prompt], sampling_params=sampling_params)
    
    # Extract semantic tokens
    predicted_tokens = outputs[0].outputs[0].text
    semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicted_tokens)
    
    if not semantic_matches:
        raise ValueError("No semantic tokens found in the generated output.")
    
    # Convert to tensors
    pred_semantic_ids = (
        torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)
    )
    pred_global_ids = torch.tensor([global_tokens]).long()
    
    # Decode to audio
    wav_np = audio_tokenizer.detokenize(
        pred_global_ids.to(device), pred_semantic_ids.to(device)
    )
    
    return wav_np


def convert_to_pcm16_bytes(audio_np: np.ndarray) -> bytes:
    """Convert numpy audio array to PCM16 bytes."""
    audio_int16 = (audio_np * 32767).astype(np.int16)
    return audio_int16.tobytes()


async def generate_audio_chunks_async(
    text: str,
    speaker_id: int,
    temperature: float
):
    """Async generator that yields audio chunks for streaming."""
    loop = asyncio.get_running_loop()
    
    try:
        # Split text into sentences
        sentences = chunk_text_simple(text)
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            print(f"Generating audio for sentence {i+1}/{len(sentences)}: {sentence[:50]}...")
            
            # Generate audio in thread pool to avoid blocking
            audio_np = await loop.run_in_executor(
                None,
                generate_audio_segment,
                sentence,
                speaker_id,
                temperature
            )
            
            # Convert to PCM bytes
            pcm_bytes = convert_to_pcm16_bytes(audio_np)
            
            if pcm_bytes:
                yield pcm_bytes
                
    except Exception as e:
        print(f"Error during audio generation: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    print("Initializing Spark TTS models...")
    initialize_models()
    print("Server ready!")


@app.websocket("/v1/audio/speech/stream/ws")
async def websocket_audio_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming audio generation.
    
    Protocol:
    - Client sends JSON: {"input": "text", "voice": "voice_name", "speaker_id": 248, "continue": true/false, "segment_id": "id"}
    - Server sends: {"type": "start", "segment_id": "id"} followed by binary audio chunks
    - Server sends: {"type": "end", "segment_id": "id"} when segment complete
    """
    await websocket.accept()
    print("WebSocket connection established")
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                text = message.get("input", "")
                voice = message.get("voice", "luganda_female")
                speaker_id = message.get("speaker_id")
                temperature = message.get("temperature", DEFAULT_TEMPERATURE)
                continue_stream = message.get("continue", True)
                segment_id = message.get("segment_id", "default")
                
                # Resolve speaker ID
                if speaker_id is None:
                    speaker_id = SPEAKER_IDS.get(voice, DEFAULT_SPEAKER_ID)
                
                if not text and not continue_stream:
                    print("Received end signal, closing stream")
                    break
                
                if text:
                    # Send start message
                    await websocket.send_json({
                        "type": "start",
                        "segment_id": segment_id,
                        "speaker_id": speaker_id
                    })
                    
                    # Stream audio chunks
                    async for audio_chunk in generate_audio_chunks_async(
                        text=text,
                        speaker_id=speaker_id,
                        temperature=temperature
                    ):
                        await websocket.send_bytes(audio_chunk)
                    
                    # Send end message
                    await websocket.send_json({
                        "type": "end",
                        "segment_id": segment_id
                    })
                    
            except WebSocketDisconnect:
                print("Client disconnected")
                break
            except json.JSONDecodeError:
                print("Invalid JSON received")
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
            except Exception as e:
                print(f"Error processing message: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("WebSocket connection closed")


@app.post("/v1/audio/speech/stream")
async def http_audio_stream(request: AudioRequest):
    """
    HTTP endpoint for streaming audio as raw PCM bytes.
    """
    print(f"Received HTTP streaming request for: '{request.text[:50]}...'")
    
    # Resolve speaker ID
    speaker_id = request.speaker_id
    if speaker_id is None:
        speaker_id = SPEAKER_IDS.get(request.voice, DEFAULT_SPEAKER_ID)
    
    async def stream_pcm():
        async for chunk in generate_audio_chunks_async(
            text=request.text,
            speaker_id=speaker_id,
            temperature=request.temperature
        ):
            yield chunk
    
    return StreamingResponse(
        stream_pcm(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(AUDIO_SAMPLERATE),
            "X-Bit-Depth": str(AUDIO_BITS_PER_SAMPLE),
            "X-Channels": str(AUDIO_CHANNELS),
        }
    )


@app.get("/")
async def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Spark TTS Streaming API",
        "model": MODEL_NAME,
        "sample_rate": AUDIO_SAMPLERATE,
        "available_voices": list(SPEAKER_IDS.keys()),
        "endpoints": {
            "websocket": "/v1/audio/speech/stream/ws",
            "http": "/v1/audio/speech/stream"
        },
        "example_usage": {
            "websocket": {
                "connect": "ws://localhost:8001/v1/audio/speech/stream/ws",
                "send": {
                    "input": "Your text here",
                    "voice": "luganda_female",
                    "segment_id": "segment_1"
                }
            },
            "http": {
                "url": "POST /v1/audio/speech/stream",
                "body": {
                    "text": "Your text here",
                    "voice": "luganda_female",
                    "temperature": 0.7
                }
            }
        }
    }


@app.get("/v1/voices")
async def list_voices():
    """List available voices."""
    return {
        "voices": [
            {"id": name, "speaker_id": sid, "language": name.split("_")[0]}
            for name, sid in SPEAKER_IDS.items()
        ]
    }


if __name__ == "__main__":
    print("Starting Spark TTS FastAPI server with WebSocket support...")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {CUDA_VISIBLE_DEVICES}")
    uvicorn.run(
        "spark_tts_streaming:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )