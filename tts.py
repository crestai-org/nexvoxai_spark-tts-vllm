from __future__ import annotations

import asyncio
import json
import weakref
import time
from dataclasses import dataclass, replace

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

AUDIO_FRAME_SIZE_MS = 20


@dataclass
class _TTSOptions:
    voice: str
    speaker_id: int | None
    temperature: float
    base_url: str

    def _normalize_url(self, url: str) -> str:
        """Ensure the URL ends with exactly one slash."""
        return url.rstrip('/') + '/'

    def get_ws_url(self, path: str) -> str:
        """Get WebSocket URL for a given path."""
        path = path.lstrip('/')
        base = self._normalize_url(self.base_url)
        # Convert http to ws or https to wss
        ws_base = base.replace('http://', 'ws://').replace('https://', 'wss://')
        return f"{ws_base}{path}"


class SparkTTS(tts.TTS):
    """
    LiveKit TTS plugin for Spark TTS real-time streaming synthesis.
    
    Supports multiple African language voices including Acholi, Ateso, 
    Runyankore, Lugbara, Swahili, and Luganda.
    
    Connects to our real-time WebSocket streaming endpoint for optimal performance.
    
    Example usage:
        from spark_tts_plugin import SparkTTS
        
        # Using voice name
        tts_instance = SparkTTS(
            base_url="http://35.203.124.213:8000/",
            voice="luganda_female",
            temperature=0.7
        )
        
        # Using speaker ID directly
        tts_instance = SparkTTS(
            base_url="http://35.203.124.213:8000/",
            speaker_id=248,
            temperature=0.7
        )
    """
    
    # Available voices mapping
    VOICES = {
        "acholi_female": 241,
        "ateso_female": 242,
        "runyankore_female": 243,
        "lugbara_female": 245,
        "swahili_male": 246,
        "luganda_female": 248,
    }
    
    def __init__(
        self,
        *,
        voice: str = "luganda_female",
        speaker_id: int | None = None,
        temperature: float = 0.7,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = "http://35.203.124.213:8000/",
    ) -> None:
        """
        Initialize Spark TTS plugin with real-time streaming support.
        
        Args:
            voice: Voice name (e.g., "luganda_female", "swahili_male")
            speaker_id: Direct speaker ID (overrides voice if provided)
            temperature: Generation temperature (0.1-1.0, default 0.7)
            http_session: Optional aiohttp session
            base_url: Spark TTS server URL
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=16000,  # Spark TTS uses 16kHz
            num_channels=1,
        )

        # Validate voice or speaker_id
        if speaker_id is None and voice not in self.VOICES:
            raise ValueError(
                f"Invalid voice '{voice}'. Available voices: {list(self.VOICES.keys())}"
            )

        self._opts = _TTSOptions(
            voice=voice,
            speaker_id=speaker_id,
            temperature=temperature,
            base_url=base_url,
        )
        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        """Connect to WebSocket endpoint."""
        session = self._ensure_session()
        url = self._opts.get_ws_url("/v1/audio/speech/stream/ws")
        return await asyncio.wait_for(session.ws_connect(url), timeout)

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Close WebSocket connection."""
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session exists."""
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def prewarm(self) -> None:
        """Prewarm the connection pool."""
        self._pool.prewarm()

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        speaker_id: NotGivenOr[int] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """
        Update TTS options dynamically.
        
        Args:
            voice: New voice name
            speaker_id: New speaker ID
            temperature: New temperature value
        """
        if is_given(voice):
            if voice not in self.VOICES:
                raise ValueError(
                    f"Invalid voice '{voice}'. Available: {list(self.VOICES.keys())}"
                )
            self._opts.voice = voice
            
        if is_given(speaker_id):
            self._opts.speaker_id = speaker_id
            
        if is_given(temperature):
            if not 0.1 <= temperature <= 1.0:
                raise ValueError("Temperature must be between 0.1 and 1.0")
            self._opts.temperature = temperature

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """
        Create a real-time streaming synthesis session using WebSocket.
        
        This is the primary method for LLM-generated text synthesis.
        
        Args:
            conn_options: Connection options
            
        Returns:
            SynthesizeStream for real-time synthesis
        """
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "ChunkedStream":
        """
        Synthesize text using HTTP endpoint for non-streaming use cases.
        
        Args:
            text: Text to synthesize
            conn_options: Connection options
            
        Returns:
            ChunkedStream for audio output
        """
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        """Close all streams and connections."""
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using the HTTP streaming endpoint."""

    def __init__(
        self, *, tts: SparkTTS, input_text: str, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: SparkTTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the HTTP streaming synthesis."""
        # Determine speaker_id
        speaker_id = self._opts.speaker_id
        if speaker_id is None:
            speaker_id = SparkTTS.VOICES.get(self._opts.voice)

        json_data = {
            "text": self._input_text,
            "voice": self._opts.voice,
            "speaker_id": speaker_id,
            "temperature": self._opts.temperature,
        }

        try:
            session = self._tts._ensure_session()
            http_url = self._opts._normalize_url(self._opts.base_url) + "v1/audio/speech/stream"
            
            async with session.post(
                http_url,
                json=json_data,
                timeout=aiohttp.ClientTimeout(
                    total=120, sock_connect=self._conn_options.timeout
                ),
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=16000,  # Spark TTS uses 16kHz
                    num_channels=1,
                    mime_type="audio/pcm",
                    frame_size_ms=AUDIO_FRAME_SIZE_MS,
                )

                async for data, _ in resp.content.iter_chunks():
                    if data:
                        output_emitter.push(data)

                output_emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError(f"Failed to connect to Spark TTS: {e}") from e


class SynthesizeStream(tts.SynthesizeStream):
    """
    Real-time streaming synthesis using WebSocket for LLM-generated text.
    
    This class is designed to work with LiveKit agents where text comes from LLM responses.
    The LLM pushes text fragments as they're generated, and this plugin converts them
    to real-time audio using our Spark TTS WebSocket streaming endpoint.
    
    Flow:
    1. LLM generates text fragments -> push_text()
    2. Agent calls flush() when LLM is done
    3. Complete text sent to Spark TTS server
    4. Server chunks by sentences and streams audio in real-time
    5. Audio chunks played back as they arrive
    """

    def __init__(self, *, tts: SparkTTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: SparkTTS = tts
        self._opts = replace(tts._opts)
        self._start_time: float | None = None

    def push_text(self, text: str) -> None:
        """
        Push LLM-generated text to be synthesized.
        
        Args:
            text: Text fragment from LLM to synthesize
        """
        if self._start_time is None:
            self._start_time = time.perf_counter()
        return super().push_text(text)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the WebSocket streaming synthesis for LLM text."""
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=16000,  # Spark TTS uses 16kHz
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
            frame_size_ms=AUDIO_FRAME_SIZE_MS,
        )

        # Determine speaker_id
        speaker_id = self._opts.speaker_id
        if speaker_id is None:
            speaker_id = SparkTTS.VOICES.get(self._opts.voice)

        async def _llm_text_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            """Task to send LLM-generated text to WebSocket for real-time streaming."""
            base_pkt = {
                "voice": self._opts.voice,
                "speaker_id": speaker_id,
                "temperature": self._opts.temperature,
            }
            
            # Collect LLM-generated text fragments for real-time streaming
            # LLM will push text fragments as they're generated
            complete_text = ""
            llm_text_received = False
            
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    # LLM finished generating text, send what we have
                    break
                
                # Accumulate LLM text fragments
                complete_text += data
                llm_text_received = True
                print(f"[Spark TTS] Received LLM text fragment: '{data[:50]}...' (total: {len(complete_text)} chars)")
            
            # Send complete LLM text for real-time streaming
            if llm_text_received and complete_text.strip():
                segment_id = utils.shortuuid()
                token_pkt = base_pkt.copy()
                token_pkt["input"] = complete_text
                token_pkt["continue"] = False  # Single request for real-time streaming
                token_pkt["segment_id"] = segment_id
                self._mark_started()
                
                print(f"[Spark TTS] Sending LLM-generated text for real-time streaming: {len(complete_text)} chars")
                await ws.send_str(json.dumps(token_pkt))
            else:
                print("[Spark TTS] No LLM text to process")

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            """Task to receive real-time audio from WebSocket."""
            segment_started = False
            first_chunk = True
            current_segment_id = None

            try:
                async for msg in ws:
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        break

                    if msg.type == aiohttp.WSMsgType.BINARY:
                        # Binary audio data - real-time streaming chunk
                        if not segment_started:
                            output_emitter.start_segment(segment_id=utils.shortuuid())
                            segment_started = True

                        if first_chunk and self._start_time:
                            ttfb = time.perf_counter() - self._start_time
                            print(f"[Spark TTS] TTFB: {ttfb*1000:.2f} ms - Real-time streaming started!")
                            first_chunk = False

                        chunk_size = len(msg.data)
                        print(f"[Spark TTS] Received real-time audio chunk: {chunk_size} bytes")
                        output_emitter.push(msg.data)

                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        # Control messages
                        data = json.loads(msg.data)
                        msg_type = data.get("type")

                        if msg_type == "start":
                            if segment_started:
                                output_emitter.end_segment()
                            
                            current_segment_id = data.get("segment_id", utils.shortuuid())
                            output_emitter.start_segment(segment_id=current_segment_id)
                            segment_started = True

                        elif msg_type == "end":
                            if segment_started:
                                output_emitter.end_segment()
                                segment_started = False
                                current_segment_id = None

                        elif msg_type == "error":
                            error_msg = data.get("message", "Unknown error")
                            print(f"[Spark TTS] Server error: {error_msg}")
                            raise APIConnectionError(f"TTS server error: {error_msg}")

            except asyncio.CancelledError:
                pass
            finally:
                if segment_started:
                    output_emitter.end_segment()

        ws = None
        try:
            ws = await self._tts._connect_ws(self._conn_options.timeout)
            tasks = [
                asyncio.create_task(_llm_text_task(ws)),
                asyncio.create_task(_recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except (aiohttp.ClientResponseError, aiohttp.ClientConnectionError) as e:
            raise APIConnectionError(f"Failed to connect to Spark TTS: {e}") from e
        except Exception as e:
            raise APIConnectionError(f"Spark TTS error: {e}") from e
        finally:
            if ws is not None and not ws.closed:
                await self._tts._close_ws(ws)
