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

# Configuration
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://0.0.0.0:8002/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "crestai/spark-tts-nexvox")
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
    
    print(f"Loading BiCodec audio tokenizer from {MODEL_NAME}...")
    try:
        # Import the model code to access BiCodecTokenizer class
        from huggingface_hub import hf_hub_download
        import importlib.util
        import sys
        
        # Download the model code file
        model_py = hf_hub_download(repo_id=MODEL_NAME, filename="modeling_spark.py")
        
        # Load the module
        spec = importlib.util.spec_from_file_location("spark_model", model_py)
        spark_module = importlib.util.module_from_spec(spec)
        sys.modules["spark_model"] = spark_module
        spec.loader.exec_module(spark_module)
        
        # Load BiCodec tokenizer - this is separate from the LLM
        audio_tokenizer = spark_module.BiCodecTokenizer.from_pretrained(MODEL_NAME)
        audio_tokenizer.device = DEVICE
        audio_tokenizer.model.to(DEVICE)
        
        if DEVICE == "cuda":
            audio_tokenizer.model = audio_tokenizer.model.half()
        
        print(f"BiCodec audio tokenizer loaded on {DEVICE}")
        
    except Exception as e:
        print(f"Error loading BiCodec tokenizer: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(
            f"Could not load BiCodec tokenizer from {MODEL_NAME}. "
            "The model should include modeling_spark.py with BiCodecTokenizer class."
        )
    
    print("Models initialized successfully")
    print(f"Note: vLLM is handling the LLM at {VLLM_BASE_URL}, we only load the audio decoder locally")


class AudioRequest(BaseModel):
    text: str
    voice: Optional[str] = None
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
        voice: Optional voice name prefix
    
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
    
    Args:
        text: Generated text containing token markers
    
    Returns:
        Tuple of (semantic_tokens, global_tokens)
    """
    # Extract semantic token IDs using regex
    semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", text)
    semantic_tokens = [int(token) for token in semantic_matches] if semantic_matches else []
    
    # Extract global token IDs using regex
    global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", text)
    global_tokens = [int(token) for token in global_matches] if global_matches else []
    
    return semantic_tokens, global_tokens


def decode_audio_chunk(
    semantic_tokens: List[int],
    global_tokens: List[int]
) -> np.ndarray:
    """
    Decode semantic and global tokens to audio waveform.
    
    Args:
        semantic_tokens: List of semantic token IDs
        global_tokens: List of global token IDs
    
    Returns:
        Audio waveform as numpy array
    """
    if not semantic_tokens:
        return np.array([], dtype=np.float32)
    
    # Convert to tensors with batch dimension
    pred_semantic_ids = torch.tensor(semantic_tokens).long().unsqueeze(0).to(DEVICE)
    
    # Handle global tokens - use defaults if empty
    if not global_tokens:
        pred_global_ids = torch.zeros((1, 1), dtype=torch.long).to(DEVICE)
    else:
        pred_global_ids = torch.tensor(global_tokens).long().unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    # Detokenize using BiCodecTokenizer
    with torch.no_grad():
        wav_np = audio_tokenizer.detokenize(
            pred_global_ids.squeeze(0),  # Shape (1, N_global)
            pred_semantic_ids            # Shape (1, N_semantic)
        )
    
    return wav_np


def apply_fade(audio_array: np.ndarray, fade_samples: int) -> np.ndarray:
    """Apply fade-in and fade-out to audio array."""
    if audio_array.size < 2 * fade_samples:
        return audio_array
    
    fade_in = np.linspace(0., 1., fade_samples)
    fade_out = np.linspace(1., 0., fade_samples)
    
    audio_array[:fade_samples] *= fade_in
    audio_array[-fade_samples:] *= fade_out
    
    return audio_array


def convert_to_pcm16_bytes(audio_array: np.ndarray, fade_ms: int = 5) -> bytes:
    """Convert audio array to raw PCM 16-bit bytes with optional fade."""
    if audio_array.size == 0:
        return b''
    
    # Apply fade
    if fade_ms > 0:
        fade_samples = int(AUDIO_SAMPLERATE * fade_ms / 1000)
        fade_samples = (fade_samples // 2) * 2
        if fade_samples > 0:
            audio_array = apply_fade(audio_array.copy(), fade_samples)
    
    # Convert to int16
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
    Async generator that yields raw PCM audio chunks for streaming.
    """
    loop = asyncio.get_running_loop()
    
    try:
        # Format prompt
        formatted_prompt = format_prompt_for_spark(text, voice)
        print(f"Formatted Prompt: {formatted_prompt[:100]}...")
        
        # Create streaming request
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
                
                # Extract tokens from accumulated text
                semantic_tokens, global_tokens = await loop.run_in_executor(
                    None, extract_tokens_from_text, accumulated_text
                )
                
                # Update global tokens if we found new ones
                if global_tokens:
                    all_global_tokens = global_tokens
                
                current_semantic_count = len(semantic_tokens)
                
                # Determine chunk size
                if not first_chunk_yielded:
                    chunk_size = INITIAL_CHUNK_SIZE_TOKENS
                else:
                    chunk_size = STREAM_CHUNK_SIZE_TOKENS
                
                # Process if we have enough new tokens
                if current_semantic_count >= processed_semantic_count + chunk_size:
                    tokens_to_process = current_semantic_count - processed_semantic_count
                    tokens_to_process = (tokens_to_process // chunk_size) * chunk_size
                    end_idx = processed_semantic_count + tokens_to_process
                    
                    if end_idx > processed_semantic_count:
                        # Get the tokens to process
                        semantic_chunk = semantic_tokens[processed_semantic_count:end_idx]
                        
                        # Decode audio
                        audio_array = await loop.run_in_executor(
                            None,
                            decode_audio_chunk,
                            semantic_chunk,
                            all_global_tokens
                        )
                        
                        # Convert to PCM bytes
                        pcm_bytes = convert_to_pcm16_bytes(audio_array, fade_ms=50)
                        if pcm_bytes:
                            yield pcm_bytes
                            first_chunk_yielded = True
                        
                        processed_semantic_count = end_idx
        
        # Process remaining tokens
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
    WebSocket endpoint for streaming audio generation.
    
    Protocol:
    - Client sends JSON: {"input": "text", "voice": "voice_name", "continue": true/false, "segment_id": "id"}
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
                voice = message.get("voice")
                continue_stream = message.get("continue", True)
                segment_id = message.get("segment_id", "default")
                
                if not text and not continue_stream:
                    print("Received end signal, closing stream")
                    break
                
                if text:
                    # Send start message
                    await websocket.send_json({"type": "start", "segment_id": segment_id})
                    
                    # Stream audio chunks
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
                    
                    # Send end message
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
    HTTP endpoint for streaming audio as raw PCM bytes.
    """
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
        "message": "Spark TTS Streaming API",
        "model": MODEL_NAME,
        "sample_rate": AUDIO_SAMPLERATE,
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
        "model_loaded": tokenizer is not None and audio_tokenizer is not None
    }


if __name__ == "__main__":
    print("Initializing Spark TTS Streaming Server...")
    print(f"Device: {DEVICE}")
    print(f"vLLM endpoint: {VLLM_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    
    initialize_models()
    
    print("\nStarting FastAPI server with WebSocket support...")
    print("Remember to start vLLM server first:")
    print(f"vllm serve {MODEL_NAME} --port 8002 --max-model-len 8192 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 --quantization fp8 --enable-chunked-prefill --enable-prefix-caching")
    
    uvicorn.run("spark_tts_streaming:app", host="0.0.0.0", port=8001, reload=False)