import os
import torch
from openai import AsyncOpenAI
from transformers import AutoTokenizer
import asyncio
import functools
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import json
import re
from typing import Optional, List
from pathlib import Path

# IMPORTANT: Add the Spark-TTS GitHub repo to sys.path for BiCodecTokenizer import
import sys
# Adjust the path if your Spark-TTS repo is cloned elsewhere
sys.path.append("Spark-TTS")  # Clone from: git clone https://github.com/SparkAudio/Spark-TTS

from sparktts.models.audio_tokenizer import BiCodecTokenizer  # Import from cloned repo

# Configuration
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://0.0.0.0:8002/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "crestai/spark-tts-nexvox")  # Fine-tuned LLM model
AUDIO_TOKENIZER_MODEL = os.environ.get("AUDIO_TOKENIZER_MODEL", "unsloth/Spark-TTS-0.5B")  # Original audio tokenizer
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "token123")

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 50
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_REPETITION_PENALTY = 1.0

AUDIO_SAMPLERATE = 16000  # BiCodec default sample rate
AUDIO_CHANNELS = 1

STREAM_CHUNK_SIZE_TOKENS = 50  # Process audio every 50 semantic tokens
INITIAL_CHUNK_SIZE_TOKENS = 20  # Smaller initial chunk for faster first audio

app = FastAPI()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize clients
client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
tokenizer = None
audio_tokenizer = None

print(f"Connecting to vLLM at {VLLM_BASE_URL}")


def initialize_models():
    """Initialize text tokenizer and BiCodec audio tokenizer."""
    global tokenizer, audio_tokenizer
    
    print(f"Loading text tokenizer from {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("Text tokenizer loaded")
    
    print(f"Loading BiCodec audio tokenizer from {AUDIO_TOKENIZER_MODEL}...")
    try:
        # Download audio tokenizer model (only BiCodec and wav2vec2, skip LLM)
        from huggingface_hub import snapshot_download
        
        audio_tokenizer_dir = snapshot_download(
            AUDIO_TOKENIZER_MODEL,
            local_dir=f"./models/{AUDIO_TOKENIZER_MODEL.split('/')[-1]}",
            allow_patterns=[
                "BiCodec/*",
                "wav2vec2-large-xlsr-53/*",
                "config.yaml",
                "*.json"
            ],
            local_dir_use_symlinks=False
        )
        
        print(f"Audio tokenizer model downloaded to: {audio_tokenizer_dir}")
        
        # Verify BiCodec files exist
        bicodec_path = Path(audio_tokenizer_dir) / "BiCodec"
        bicodec_model = bicodec_path / "model.safetensors"
        bicodec_config = bicodec_path / "config.yaml"
        
        if not bicodec_path.exists():
            raise FileNotFoundError(f"BiCodec directory not found at {bicodec_path}")
        if not bicodec_model.exists():
            raise FileNotFoundError(f"BiCodec model weights not found at {bicodec_model}")
        if not bicodec_config.exists():
            raise FileNotFoundError(f"BiCodec config not found at {bicodec_config}")
        
        print(f"BiCodec files verified:")
        print(f"  - Model: {bicodec_model}")
        print(f"  - Config: {bicodec_config}")
        
        # Initialize BiCodecTokenizer
        # BiCodecTokenizer expects the model directory path
        audio_tokenizer = BiCodecTokenizer(
            model_dir=audio_tokenizer_dir,
            device=DEVICE
        )
        
        if audio_tokenizer is None:
            raise RuntimeError("BiCodecTokenizer initialization returned None")
        
        print(f"✓ BiCodec audio tokenizer loaded successfully on {DEVICE}")
        
    except Exception as e:
        print(f"ERROR: Failed to load BiCodec tokenizer: {e}")
        import traceback
        traceback.print_exc()
        
        # Print diagnostic info
        print("\n" + "="*80)
        print("DIAGNOSTIC INFORMATION")
        print("="*80)
        if 'audio_tokenizer_dir' in locals():
            print(f"Download directory: {audio_tokenizer_dir}")
            bicodec_dir = Path(audio_tokenizer_dir) / "BiCodec"
            print(f"BiCodec directory exists: {bicodec_dir.exists()}")
            if bicodec_dir.exists():
                print(f"BiCodec contents: {list(bicodec_dir.iterdir())}")
        print("="*80 + "\n")
        
        raise RuntimeError(
            f"Could not load BiCodec tokenizer from {AUDIO_TOKENIZER_MODEL}. "
            "See error details above."
        )
    
    print("Models initialized successfully")
    print(f"Architecture: vLLM serves fine-tuned LLM ({MODEL_NAME}), "
          f"local BiCodec decodes audio ({AUDIO_TOKENIZER_MODEL})")


class AudioRequest(BaseModel):
    text: str
    voice: Optional[str] = None  # e.g., "248" for Luganda female
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY


def format_prompt_for_spark(text: str, voice: Optional[str] = None) -> str:
    """
    Format the text prompt for Spark TTS.
    
    Args:
        text: The text to convert to speech
        voice: Optional speaker ID prefix (e.g., "248" for Luganda female)
               Available voices:
               - 241: Acholi (female)
               - 242: Ateso (female)
               - 243: Runyankore (female)
               - 245: Lugbara (female)
               - 246: Swahili (male)
               - 248: Luganda (female)
    
    Returns:
        Formatted prompt string
    """
    if voice:
        text = f"{voice}: {text}"
    
    prompt = "".join([
        "<|task_tts|>",
        "<|start_content|>",
        text,
        "<|end_content|>",
        "<|start_global_token|>"
    ])
    
    return prompt


def extract_tokens_from_text(text: str) -> tuple[List[int], List[int]]:
    """
    Extract semantic and global tokens from generated text.
    
    Returns:
        Tuple of (semantic_tokens, global_tokens)
    """
    semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", text)
    semantic_tokens = [int(token) for token in semantic_matches] if semantic_matches else []
    
    global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", text)
    global_tokens = [int(token) for token in global_matches] if global_matches else []
    
    return semantic_tokens, global_tokens


def decode_audio_chunk(
    semantic_tokens: List[int],
    global_tokens: List[int]
) -> np.ndarray:
    """
    Decode semantic and global tokens to audio waveform using BiCodecTokenizer.
    
    Args:
        semantic_tokens: List of semantic token IDs
        global_tokens: List of global token IDs
    
    Returns:
        Audio waveform as numpy array
    """
    if not semantic_tokens:
        return np.array([], dtype=np.float32)
    
    pred_semantic_ids = torch.tensor(semantic_tokens).long().unsqueeze(0).to(DEVICE)
    
    if not global_tokens:
        pred_global_ids = torch.zeros((1, 1), dtype=torch.long).to(DEVICE)
    else:
        pred_global_ids = torch.tensor(global_tokens).long().unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        wav_np = audio_tokenizer.detokenize(
            pred_global_ids.squeeze(0),  # (N_global,)
            pred_semantic_ids            # (1, N_semantic)
        )
    
    return wav_np


def apply_fade(audio_array: np.ndarray, fade_samples: int) -> np.ndarray:
    """Apply fade-in and fade-out to reduce clicks."""
    if audio_array.size < 2 * fade_samples:
        return audio_array
    
    fade_in = np.linspace(0., 1., fade_samples)
    fade_out = np.linspace(1., 0., fade_samples)
    
    audio_array[:fade_samples] *= fade_in
    audio_array[-fade_samples:] *= fade_out
    
    return audio_array


def convert_to_pcm16_bytes(audio_array: np.ndarray, fade_ms: int = 5) -> bytes:
    """
    Convert float audio array to raw PCM 16-bit bytes with optional fade.
    
    Args:
        audio_array: Float audio array (range -1 to 1)
        fade_ms: Fade duration in milliseconds
    
    Returns:
        Raw PCM 16-bit bytes
    """
    if audio_array.size == 0:
        return b''
    
    if fade_ms > 0:
        fade_samples = int(AUDIO_SAMPLERATE * fade_ms / 1000)
        fade_samples = (fade_samples // 2) * 2
        if fade_samples > 0:
            audio_array = apply_fade(audio_array.copy(), fade_samples)
    
    audio_int16 = (audio_array * 32767).clip(-32768, 32767).astype(np.int16)
    return audio_int16.tobytes()


async def generate_audio_chunks(
    text: str,
    voice: Optional[str],
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    repetition_penalty: float
):
    """
    Async generator yielding raw PCM audio chunks for streaming.
    
    This function:
    1. Formats the prompt for Spark TTS
    2. Streams token generation from vLLM
    3. Extracts semantic/global tokens from the output
    4. Decodes tokens to audio in chunks for low-latency streaming
    """
    loop = asyncio.get_running_loop()
    
    try:
        formatted_prompt = format_prompt_for_spark(text, voice)
        print(f"Formatted Prompt: {formatted_prompt[:100]}...")
        
        stream_kwargs = dict(
            model=MODEL_NAME,
            prompt=formatted_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            extra_body={
                'repetition_penalty': repetition_penalty,
                'top_k': top_k
            },
        )
        
        response_stream = await client.completions.create(**stream_kwargs)
        
        accumulated_text = ""
        processed_semantic_count = 0
        all_global_tokens = []
        first_chunk_yielded = False
        
        async for chunk in response_stream:
            if chunk.choices:
                chunk_text = chunk.choices[0].text or ""
                accumulated_text += chunk_text
                
                semantic_tokens, global_tokens = await loop.run_in_executor(
                    None, extract_tokens_from_text, accumulated_text
                )
                
                if global_tokens:
                    all_global_tokens = global_tokens
                
                current_semantic_count = len(semantic_tokens)
                
                # Use smaller chunk for first audio output (lower latency)
                chunk_size = INITIAL_CHUNK_SIZE_TOKENS if not first_chunk_yielded else STREAM_CHUNK_SIZE_TOKENS
                
                if current_semantic_count >= processed_semantic_count + chunk_size:
                    tokens_to_process = (current_semantic_count - processed_semantic_count) // chunk_size * chunk_size
                    end_idx = processed_semantic_count + tokens_to_process
                    
                    if end_idx > processed_semantic_count:
                        semantic_chunk = semantic_tokens[processed_semantic_count:end_idx]
                        
                        audio_array = await loop.run_in_executor(
                            None,
                            decode_audio_chunk,
                            semantic_chunk,
                            all_global_tokens
                        )
                        
                        pcm_bytes = convert_to_pcm16_bytes(audio_array, fade_ms=50)
                        if pcm_bytes:
                            yield pcm_bytes
                            first_chunk_yielded = True
                        
                        processed_semantic_count = end_idx
        
        # Process remaining tokens after stream ends
        semantic_tokens, global_tokens = await loop.run_in_executor(
            None, extract_tokens_from_text, accumulated_text
        )
        
        if global_tokens:
            all_global_tokens = global_tokens
        
        if len(semantic_tokens) > processed_semantic_count:
            remaining_tokens = semantic_tokens[processed_semantic_count:]
            if remaining_tokens:
                audio_array = await loop.run_in_executor(
                    None,
                    decode_audio_chunk,
                    remaining_tokens,
                    all_global_tokens
                )
                
                pcm_bytes = convert_to_pcm16_bytes(audio_array, fade_ms=50)
                if pcm_bytes:
                    yield pcm_bytes
        
        print(f"Generated {len(semantic_tokens)} semantic tokens, {len(all_global_tokens)} global tokens")
        
    except Exception as e:
        print(f"Error during audio generation: {e}")
        import traceback
        traceback.print_exc()


@app.websocket("/v1/audio/speech/stream/ws")
async def websocket_audio_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS with bi-directional communication.
    
    Client sends JSON messages with:
    - input: Text to synthesize
    - voice: Optional speaker ID (e.g., "248")
    - segment_id: Optional ID to track segments
    - continue: Set to false to end stream
    
    Server sends:
    - JSON start message
    - Binary audio chunks (raw PCM)
    - JSON end message
    """
    # Check if models are loaded before accepting connection
    if tokenizer is None or audio_tokenizer is None:
        await websocket.close(code=1011, reason="Models not initialized")
        return
    
    await websocket.accept()
    print("WebSocket connection established")
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                text = message.get("input", "")
                voice = message.get("voice")
                continue_stream = message.get("continue", True)
                segment_id = message.get("segment_id", "default")
                
                if not text and not continue_stream:
                    print("Received end signal, closing stream")
                    break
                
                if text:
                    await websocket.send_json({"type": "start", "segment_id": segment_id})
                    
                    async for audio_chunk in generate_audio_chunks(
                        text=text,
                        voice=voice,
                        temperature=DEFAULT_TEMPERATURE,
                        top_p=DEFAULT_TOP_P,
                        top_k=DEFAULT_TOP_K,
                        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                        repetition_penalty=DEFAULT_REPETITION_PENALTY
                    ):
                        await websocket.send_bytes(audio_chunk)
                    
                    await websocket.send_json({"type": "end", "segment_id": segment_id})
                    
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
    HTTP endpoint for streaming TTS.
    
    Returns raw PCM audio stream with headers indicating format.
    """
    # Check if models are loaded
    if tokenizer is None or audio_tokenizer is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503, 
            detail="Models not initialized. Check server logs."
        )
    
    print(f"Received HTTP streaming request for: '{request.text[:50]}...'")
    
    async def stream_pcm():
        async for chunk in generate_audio_chunks(
            text=request.text,
            voice=request.voice,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_new_tokens=request.max_new_tokens,
            repetition_penalty=request.repetition_penalty
        ):
            yield chunk
    
    return StreamingResponse(
        stream_pcm(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(AUDIO_SAMPLERATE),
            "X-Bit-Depth": "16",
            "X-Channels": str(AUDIO_CHANNELS)
        }
    )


@app.get("/")
async def read_root():
    return {
        "message": "Spark TTS Streaming API (Fine-tuned for East African Languages)",
        "llm_model": MODEL_NAME,
        "audio_tokenizer": AUDIO_TOKENIZER_MODEL,
        "sample_rate": AUDIO_SAMPLERATE,
        "available_voices": {
            "241": "Acholi (female)",
            "242": "Ateso (female)",
            "243": "Runyankore (female)",
            "245": "Lugbara (female)",
            "246": "Swahili (male)",
            "248": "Luganda (female)"
        },
        "endpoints": {
            "websocket": "/v1/audio/speech/stream/ws",
            "http": "/v1/audio/speech/stream"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": DEVICE,
        "llm_model": MODEL_NAME,
        "audio_tokenizer": AUDIO_TOKENIZER_MODEL,
        "models_loaded": tokenizer is not None and audio_tokenizer is not None
    }


if __name__ == "__main__":
    print("="*80)
    print("Initializing Spark TTS Streaming Server")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"vLLM endpoint: {VLLM_BASE_URL}")
    print(f"Fine-tuned LLM Model: {MODEL_NAME}")
    print(f"Audio Tokenizer Model: {AUDIO_TOKENIZER_MODEL}")
    print("="*80)
    
    initialize_models()
    
    print("\n" + "="*80)
    print("Starting FastAPI server with WebSocket support...")
    print("="*80)
    print("IMPORTANT: Ensure vLLM server is running first:")
    print(f"  vllm serve {MODEL_NAME} \\")
    print(f"    --port 8002 \\")
    print(f"    --max-model-len 8192 \\")
    print(f"    --gpu-memory-utilization 0.85 \\")
    print(f"    --quantization fp8 \\")
    print(f"    --enable-chunked-prefill \\")
    print(f"    --enable-prefix-caching")
    print("="*80 + "\n")
    
    uvicorn.run("spark_tts_streaming:app", host="0.0.0.0", port=8000, reload=False)