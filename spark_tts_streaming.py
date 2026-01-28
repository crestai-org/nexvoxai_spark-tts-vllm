import os
import sys
import torch
import numpy as np
import re
import json
import asyncio
import time
import librosa
import tempfile
import soundfile as sf
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from vllm import LLM
from vllm.sampling_params import SamplingParams
from huggingface_hub import snapshot_download

# Configuration
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
MODEL_NAME = os.environ.get("MODEL_NAME", "crestai/spark-tts-nexvox-v4")
TOKENIZER_REPO = os.environ.get("TOKENIZER_REPO", "unsloth/Spark-TTS-0.5B")
TOKENIZER_CACHE_DIR = os.environ.get("TOKENIZER_CACHE_DIR", "Spark-TTS-0.5B")
SPARK_TTS_REPO_PATH = os.environ.get("SPARK_TTS_REPO_PATH", "Spark-TTS")

# Set NCCL environment variables for multi-GPU communication
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_NET_GDR_LEVEL"] = "0"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["NCCL_TREE_THRESHOLD"] = "0"
os.environ["NCCL_RING_THRESHOLD"] = "8388608"

# Audio configuration
AUDIO_SAMPLERATE = 16000
AUDIO_BITS_PER_SAMPLE = 16
AUDIO_CHANNELS = 1

# Default parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_SPEAKER_ID = 243  # Runyankore female

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
    voice: str = "runyankore_female"
    speaker_id: Optional[int] = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS


class VoiceCloningRequest(BaseModel):
    text: str
    reference_audio_path: str
    reference_text: Optional[str] = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS


def chunk_text(text: str, max_chunk_size: int = 500) -> List[str]:
    """
    Split text into chunks based on sentence boundaries.
    
    This approach preserves natural sentence flow and intonation for TTS.
    
    Args:
        text: The input string to chunk
        max_chunk_size: Maximum character length per chunk (soft limit)
    
    Returns:
        List of text chunks, each containing one or more complete sentences
    """
    # Split on sentence-ending punctuation (. ! ?) followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_length = len(sentence)
        
        # Start new chunk if adding this sentence would exceed limit
        if current_chunk and (current_length + sentence_length + 1) > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length + 1
    
    # Add the final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def chunk_text_simple(text: str) -> List[str]:
    """
    Split text into individual sentences.
    
    Recommended for TTS - provides maximum control with one sentence per chunk.
    
    Args:
        text: The input string to chunk
    
    Returns:
        List of individual sentences
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_text_with_count(text: str, sentences_per_chunk: int = 3) -> List[str]:
    """
    Split text into chunks containing a specific number of sentences.
    
    Args:
        text: The input string to chunk
        sentences_per_chunk: Number of sentences to include in each chunk
    
    Returns:
        List of text chunks
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks: List[str] = []
    
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ' '.join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
    
    return chunks


def extract_speaker_from_reference(
    audio_path: str,
    audio_tokenizer,
    reference_text: str = None,
    device="cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract global and semantic tokens from a reference audio file.
    Returns: (global_ids, semantic_ids)
    """
    # Load audio and resample to 16kHz if needed
    wav, sr = sf.read(audio_path)
    
    # Resample if not 16kHz
    if sr != 16000:
        print(f"Resampling audio from {sr}Hz to 16000Hz...")
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        
        # Save resampled audio to a temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)  # Close the file descriptor
        sf.write(temp_path, wav, 16000)
        audio_path_to_use = temp_path
    else:
        audio_path_to_use = audio_path
    
    try:
        # Tokenize reference audio using the file path
        global_ids, semantic_ids = audio_tokenizer.tokenize(audio_path_to_use)
        
        # Convert to tensors if they aren't already
        if not isinstance(global_ids, torch.Tensor):
            global_ids = torch.tensor(global_ids).long()
        if not isinstance(semantic_ids, torch.Tensor):
            semantic_ids = torch.tensor(semantic_ids).long()
        
        # Ensure they're 1D tensors
        if global_ids.dim() > 1:
            global_ids = global_ids.squeeze()
        if semantic_ids.dim() > 1:
            semantic_ids = semantic_ids.squeeze()
            
        return global_ids, semantic_ids
        
    finally:
        # Clean up temporary file if we created one
        if sr != 16000 and os.path.exists(temp_path):
            os.unlink(temp_path)


def text_to_speech_cloned(
    text: str,
    audio_tokenizer,
    model,
    reference_audio_path: str,
    reference_text: str = None,
    temperature: float = 0.7,
    device="cuda"
):
    '''Create a wav array using zero-shot voice cloning from reference audio.'''
    texts = chunk_text_simple(text)
    texts = [t.strip() for t in texts if len(t.strip()) > 0]
    
    sampling_params = SamplingParams(temperature=temperature, max_tokens=2048)
    
    # 1. Extract speaker identity from reference
    print("Extracting speaker features from reference audio...")
    global_ids_ref, semantic_ids_ref = extract_speaker_from_reference(
        reference_audio_path, audio_tokenizer, reference_text, device
    )
    
    # Convert to list for prompt formatting
    global_ids_list = global_ids_ref.cpu().tolist()
    if isinstance(global_ids_list, int):
        global_ids_list = [global_ids_list]
    
    print(f"Extracted {len(global_ids_list)} global tokens from reference")
    
    prompts = []
    for chunk in texts:
        # Build prompt with reference global tokens
        prompt = f"<|task_tts|><|start_content|>{chunk}<|end_content|><|start_global_token|>"
        prompt += ''.join([f'<|bicodec_global_{t}|>' for t in global_ids_list]) 
        prompt += '<|end_global_token|><|start_semantic_token|>'
        prompts.append(prompt)
    
    print("Generating speech...")
    outputs = model.generate(
        prompts=prompts,
        sampling_params=sampling_params
    )
    
    speech_segments = []
    
    for i in range(len(outputs)):
        predicted_tokens = outputs[i].outputs[0].text
        semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicted_tokens)
        if not semantic_matches:
            raise ValueError("No semantic tokens found in output.")
        
        pred_semantic_ids = torch.tensor([int(t) for t in semantic_matches]).long().unsqueeze(0)
        pred_global_ids = torch.tensor([global_ids_list]).long()
        
        wav_np = audio_tokenizer.detokenize(
            pred_global_ids.to(device),
            pred_semantic_ids.to(device)
        )
        speech_segments.append(wav_np)
    
    result_wav = np.concatenate(speech_segments)
    return result_wav


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
    
    # Load vLLM model with single GPU to avoid tensor parallelism issues
    print(f"Loading Spark TTS model: {MODEL_NAME}...")
    vllm_model = LLM(
        MODEL_NAME,
        enforce_eager=False,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1  # Use single GPU to avoid multi-GPU issues
    )
    print("✅ Model loaded successfully!")
    
    # Download tokenizer if needed
    if not os.path.exists(TOKENIZER_CACHE_DIR) or not os.path.exists(f"{TOKENIZER_CACHE_DIR}/config.yaml"):
        print(f"Downloading tokenizer from {TOKENIZER_REPO}...")
        snapshot_download(
            repo_id=TOKENIZER_REPO,
            local_dir=TOKENIZER_CACHE_DIR,
        )
        print(f"✅ Tokenizer downloaded to {TOKENIZER_CACHE_DIR}")
    else:
        print(f"✅ Tokenizer already exists at {TOKENIZER_CACHE_DIR}")
    
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
    """Generate audio for a single text segment with strict memory limits."""
    global_tokens = GLOBAL_IDS_BY_SPEAKER[speaker_id]
    
    # Clear CUDA cache before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
    
    # Strict semantic token limit to prevent memory issues
    if len(semantic_matches) > 800:
        semantic_matches = semantic_matches[:800]
        print(f"Limited semantic tokens to 800 for memory safety")
    
    # Convert to tensors
    pred_semantic_ids = (
        torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)
    )
    pred_global_ids = torch.tensor([global_tokens]).long()
    
    # Decode to audio with aggressive memory management
    with torch.no_grad():  # Disable gradient computation
        wav_np = audio_tokenizer.detokenize(
            pred_global_ids.to(device), pred_semantic_ids.to(device)
        )
    
    # Aggressive cleanup
    del pred_semantic_ids, pred_global_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all CUDA operations complete
        # Force garbage collection
        import gc
        gc.collect()
    
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
            
            print(f"Generated audio shape: {audio_np.shape}, PCM bytes length: {len(pcm_bytes)}")
            
            if len(pcm_bytes) > 0:
                # Yield immediately for real-time streaming
                yield pcm_bytes
                print(f"Streamed audio chunk {i+1}/{len(sentences)} immediately")
            else:
                print("Warning: Generated empty audio data")
                
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
    
    # Set up ping/pong for connection health monitoring
    ping_task = None
    last_activity = time.time()
    
    async def ping_loop():
        """Send periodic pings to keep connection alive."""
        nonlocal last_activity
        while True:
            try:
                await asyncio.sleep(30)  # Ping every 30 seconds
                if time.time() - last_activity > 60:  # No activity for 1 minute
                    print("Connection idle, sending ping")
                    await websocket.send_json({"type": "ping"})
                    last_activity = time.time()
            except Exception:
                break
    
    try:
        ping_task = asyncio.create_task(ping_loop())
        
        while True:
            try:
                # Set a reasonable timeout for receiving messages
                data = await asyncio.wait_for(websocket.receive_text(), timeout=300)  # 5 minutes
                last_activity = time.time()
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
                    
                    # Stream audio chunks continuously for the session
                    chunk_count = 0
                    try:
                        async_generator = generate_audio_chunks_async(
                            text=text,
                            speaker_id=speaker_id,
                            temperature=temperature
                        )
                        
                        # Process all chunks without timeout - let the session stay open
                        async for audio_chunk in async_generator:
                            chunk_count += 1
                            print(f"Immediately sending audio chunk {chunk_count}: {len(audio_chunk)} bytes")
                            await websocket.send_bytes(audio_chunk)
                            print(f"Audio chunk {chunk_count} sent and played immediately")
                            last_activity = time.time()
                                
                    except Exception as e:
                        print(f"Error during audio generation: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Audio generation error: {str(e)}"
                        })
                        continue
                    
                    print(f"Finished streaming {chunk_count} audio chunks, sending end message")
                    # Send end message but keep session open for more sentences
                    await websocket.send_json({
                        "type": "end",
                        "segment_id": segment_id
                    })
                    
            except asyncio.TimeoutError:
                print("Client timeout, closing connection")
                break
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
        if ping_task:
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass
        print("WebSocket connection closed")


@app.websocket("/v1/audio/speech/clone/ws")
async def websocket_voice_cloning(websocket: WebSocket):
    """
    WebSocket endpoint for voice cloning streaming.
    
    Protocol:
    - Client sends JSON: {"input": "text", "reference_audio_path": "path/to/audio.wav", "reference_text": "optional", "temperature": 0.7, "segment_id": "id"}
    - Server sends: {"type": "start", "segment_id": "id"} followed by binary audio chunks
    - Server sends: {"type": "end", "segment_id": "id"} when segment complete
    """
    await websocket.accept()
    print("Voice cloning WebSocket connection established")
    
    # Set up ping/pong for connection health monitoring
    ping_task = None
    last_activity = time.time()
    
    async def ping_loop():
        """Send periodic pings to keep connection alive."""
        nonlocal last_activity
        while True:
            try:
                await asyncio.sleep(30)  # Ping every 30 seconds
                if time.time() - last_activity > 60:  # No activity for 1 minute
                    print("Connection idle, sending ping")
                    await websocket.send_json({"type": "ping"})
                    last_activity = time.time()
            except Exception:
                break
    
    try:
        ping_task = asyncio.create_task(ping_loop())
        
        while True:
            try:
                # Set a reasonable timeout for receiving messages
                data = await asyncio.wait_for(websocket.receive_text(), timeout=300)  # 5 minutes
                last_activity = time.time()
                message = json.loads(data)
                
                text = message.get("input", "")
                reference_audio_path = message.get("reference_audio_path", "")
                reference_text = message.get("reference_text")
                temperature = message.get("temperature", DEFAULT_TEMPERATURE)
                segment_id = message.get("segment_id", "default")
                
                if not text:
                    print("Received empty text, skipping")
                    continue
                
                if not reference_audio_path:
                    await websocket.send_json({
                        "type": "error",
                        "message": "reference_audio_path is required for voice cloning"
                    })
                    continue
                
                # Validate reference audio file exists
                if not os.path.exists(reference_audio_path):
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Reference audio file not found: {reference_audio_path}"
                    })
                    continue
                
                # Send start message
                await websocket.send_json({
                    "type": "start",
                    "segment_id": segment_id,
                    "reference_audio": reference_audio_path
                })
                
                # Generate cloned audio
                try:
                    loop = asyncio.get_running_loop()
                    
                    # Generate cloned audio in thread pool
                    result_wav = await loop.run_in_executor(
                        None,
                        text_to_speech_cloned,
                        text,
                        audio_tokenizer,
                        vllm_model,
                        reference_audio_path,
                        reference_text,
                        temperature,
                        device
                    )
                    
                    # Convert to PCM bytes
                    pcm_bytes = convert_to_pcm16_bytes(result_wav)
                    
                    print(f"Generated cloned audio: {len(result_wav)} samples, {len(pcm_bytes)} bytes")
                    
                    if len(pcm_bytes) > 0:
                        # Send audio data
                        await websocket.send_bytes(pcm_bytes)
                        print(f"Cloned audio sent for segment {segment_id}")
                    else:
                        print("Warning: Generated empty cloned audio data")
                        
                except Exception as e:
                    print(f"Error during voice cloning: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Voice cloning error: {str(e)}"
                    })
                    continue
                
                # Send end message
                await websocket.send_json({
                    "type": "end",
                    "segment_id": segment_id
                })
                    
            except asyncio.TimeoutError:
                print("Client timeout, closing connection")
                break
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
        if ping_task:
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass
        print("Voice cloning WebSocket connection closed")


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


@app.post("/v1/audio/speech/clone")
async def voice_cloning_http(request: VoiceCloningRequest):
    """
    HTTP endpoint for voice cloning using reference audio.
    
    Args:
        request: VoiceCloningRequest containing text, reference audio path, and parameters
    
    Returns:
        StreamingResponse with PCM audio data
    """
    print(f"Received voice cloning request for: '{request.text[:50]}...'")
    print(f"Reference audio: {request.reference_audio_path}")
    
    # Validate reference audio file exists
    if not os.path.exists(request.reference_audio_path):
        return {"error": f"Reference audio file not found: {request.reference_audio_path}"}
    
    async def stream_cloned_pcm():
        loop = asyncio.get_running_loop()
        
        try:
            # Generate cloned audio in thread pool
            result_wav = await loop.run_in_executor(
                None,
                text_to_speech_cloned,
                request.text,
                audio_tokenizer,
                vllm_model,
                request.reference_audio_path,
                request.reference_text,
                request.temperature,
                device
            )
            
            # Convert to PCM bytes
            pcm_bytes = convert_to_pcm16_bytes(result_wav)
            
            print(f"Generated cloned audio: {len(result_wav)} samples, {len(pcm_bytes)} bytes")
            
            if len(pcm_bytes) > 0:
                yield pcm_bytes
            else:
                print("Warning: Generated empty cloned audio data")
                
        except Exception as e:
            print(f"Error during voice cloning: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    return StreamingResponse(
        stream_cloned_pcm(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(AUDIO_SAMPLERATE),
            "X-Bit-Depth": str(AUDIO_BITS_PER_SAMPLE),
            "X-Channels": str(AUDIO_CHANNELS),
            "X-Voice-Cloning": "true"
        }
    )


@app.get("/")
async def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Spark TTS Streaming API with Voice Cloning",
        "model": MODEL_NAME,
        "sample_rate": AUDIO_SAMPLERATE,
        "available_voices": list(SPEAKER_IDS.keys()),
        "features": ["text-to-speech", "voice-cloning", "streaming"],
        "endpoints": {
            "websocket": "/v1/audio/speech/stream/ws",
            "http": "/v1/audio/speech/stream",
            "voice_cloning_websocket": "/v1/audio/speech/clone/ws",
            "voice_cloning_http": "/v1/audio/speech/clone",
            "voices": "/v1/voices"
        },
        "example_usage": {
            "websocket": {
                "connect": "ws://localhost:8000/v1/audio/speech/stream/ws",
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
            },
            "voice_cloning_websocket": {
                "connect": "ws://localhost:8000/v1/audio/speech/clone/ws",
                "send": {
                    "input": "Your text here",
                    "reference_audio_path": "/path/to/reference.wav",
                    "reference_text": "Optional transcript of reference audio",
                    "temperature": 0.7,
                    "segment_id": "clone_segment_1"
                }
            },
            "voice_cloning_http": {
                "url": "POST /v1/audio/speech/clone",
                "body": {
                    "text": "Your text here",
                    "reference_audio_path": "/path/to/reference.wav",
                    "reference_text": "Optional transcript of reference audio",
                    "temperature": 0.7
                }
            }
        },
        "voice_cloning_info": {
            "description": "Zero-shot voice cloning using reference audio",
            "requirements": [
                "Reference audio file must exist and be accessible",
                "Audio will be automatically resampled to 16kHz",
                "Supports WAV, MP3, and other audio formats"
            ],
            "parameters": {
                "text": "Text to synthesize (required)",
                "reference_audio_path": "Path to reference audio file (required)",
                "reference_text": "Optional transcript of reference audio",
                "temperature": "Controls randomness (0.0-1.0, default: 0.7)"
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
        reload=False,
        ws_ping_interval=20,
        ws_ping_timeout=20,
        timeout_keep_alive=300
    )