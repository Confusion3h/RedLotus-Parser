"""
STT Engine — faster-whisper for batch files + microphone streaming for live mode.

Supports optional NVIDIA Riva ASR via gRPC when configured.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default capture settings for live mic
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SECONDS = 2.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)


@dataclass
class TranscriptChunk:
    """A piece of transcribed text with timing metadata."""

    text: str
    start: float = 0.0
    end: float = 0.0
    is_final: bool = True


@dataclass
class STTConfig:
    model_size: str = "medium"
    device: str = "auto"  # auto | cpu | cuda
    compute_type: str = "auto"  # auto | int8 | float16 | float32
    language: str = "it"
    beam_size: int = 5
    # Optional Riva endpoint, e.g. "localhost:50051"
    riva_uri: Optional[str] = None
    sample_rate: int = SAMPLE_RATE
    chunk_seconds: float = CHUNK_SECONDS


class STTEngine:
    """
    Local speech-to-text engine.

    - ``transcribe_file``: batch transcription of .mp3 / .wav / .m4a
    - ``stream_microphone``: real-time mic capture → chunked transcription
    """

    def __init__(
        self,
        config: Optional[STTConfig] = None,
        load_model: bool = True,
    ) -> None:
        self.config = config or STTConfig()
        self._model = None
        self._backend: str = "faster-whisper"
        if load_model:
            self._load_model()
        else:
            self._backend = "stub"
            logger.info("STTEngine started in stub mode (load_model=False)")

    # ------------------------------------------------------------------ load
    def _resolve_device_and_type(self) -> tuple[str, str]:
        device = self.config.device
        compute_type = self.config.compute_type

        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        return device, compute_type

    def _load_model(self) -> None:
        """Load faster-whisper (default) or fall back to a stub for offline scaffolding."""
        if self.config.riva_uri:
            self._backend = "riva"
            logger.info("Riva ASR configured at %s (lazy client)", self.config.riva_uri)
            return

        try:
            from faster_whisper import WhisperModel

            device, compute_type = self._resolve_device_and_type()
            logger.info(
                "Loading faster-whisper model=%s device=%s compute=%s",
                self.config.model_size,
                device,
                compute_type,
            )
            self._model = WhisperModel(
                self.config.model_size,
                device=device,
                compute_type=compute_type,
            )
            self._backend = "faster-whisper"
            logger.info("faster-whisper ready")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "faster-whisper unavailable (%s). Using stub STT for scaffolding.",
                exc,
            )
            self._model = None
            self._backend = "stub"

    # ----------------------------------------------------------- batch file
    def transcribe_file(self, audio_path: str | Path) -> str:
        """Transcribe a recorded audio file. Returns full Italian transcript."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if self._backend == "riva":
            return self._transcribe_file_riva(path)

        if self._backend == "stub" or self._model is None:
            logger.warning("Stub STT: returning placeholder transcript for %s", path.name)
            return (
                f"[STUB] Trascrizione simulata del file {path.name}. "
                "Mario Rossi, CF RSSMRA80A01H501U, IBAN IT60X0542811101000000123456."
            )

        segments, info = self._model.transcribe(
            str(path),
            language=self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=True,
        )
        texts = [seg.text.strip() for seg in segments if seg.text.strip()]
        transcript = " ".join(texts)
        logger.info(
            "Transcribed %s (lang=%s, duration≈%.1fs) → %d chars",
            path.name,
            getattr(info, "language", self.config.language),
            getattr(info, "duration", 0.0),
            len(transcript),
        )
        return transcript

    def _transcribe_file_riva(self, path: Path) -> str:
        """Optional NVIDIA Riva offline recognition (requires nvidia-riva-client)."""
        try:
            import riva.client  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Riva backend selected but nvidia-riva-client is not installed"
            ) from exc

        auth = riva.client.Auth(uri=self.config.riva_uri)
        asr = riva.client.ASRService(auth)
        config = riva.client.RecognitionConfig(
            language_code="it-IT",
            max_alternatives=1,
            enable_automatic_punctuation=True,
            sample_rate_hertz=self.config.sample_rate,
        )
        with open(path, "rb") as fh:
            audio_bytes = fh.read()
        response = asr.offline_recognize(audio_bytes, config)
        parts = []
        for result in response.results:
            if result.alternatives:
                parts.append(result.alternatives[0].transcript.strip())
        return " ".join(parts)

    # ----------------------------------------------------------- live mic
    def stream_microphone(
        self,
        duration_seconds: Optional[float] = None,
        on_chunk: Optional[Callable[[TranscriptChunk], None]] = None,
    ) -> Generator[TranscriptChunk, None, None]:
        """
        Capture microphone audio and yield transcript chunks in near real-time.

        Uses sounddevice when available, otherwise PyAudio.
        If ``duration_seconds`` is set, stops after that many seconds.
        """
        audio_q: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        stop_event = threading.Event()
        started = time.monotonic()

        def _producer_sounddevice() -> None:
            import sounddevice as sd

            blocksize = int(self.config.sample_rate * self.config.chunk_seconds)

            def callback(indata, frames, time_info, status):  # noqa: ARG001
                if status:
                    logger.debug("sounddevice status: %s", status)
                audio_q.put(indata.copy().reshape(-1).astype(np.float32))

            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=CHANNELS,
                dtype="float32",
                blocksize=blocksize,
                callback=callback,
            ):
                while not stop_event.is_set():
                    time.sleep(0.05)

        def _producer_pyaudio() -> None:
            import pyaudio

            pa = pyaudio.PyAudio()
            frames_per_buffer = int(self.config.sample_rate * self.config.chunk_seconds)
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=frames_per_buffer,
            )
            try:
                while not stop_event.is_set():
                    raw = stream.read(frames_per_buffer, exception_on_overflow=False)
                    audio = (
                        np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    )
                    audio_q.put(audio)
            finally:
                stream.stop_stream()
                stream.close()
                pa.terminate()

        backend_name = "sounddevice"
        try:
            import sounddevice  # noqa: F401
            producer = threading.Thread(target=_producer_sounddevice, daemon=True)
        except Exception:
            backend_name = "pyaudio"
            producer = threading.Thread(target=_producer_pyaudio, daemon=True)

        logger.info("Live mic capture via %s (chunk=%.1fs)", backend_name, self.config.chunk_seconds)
        producer.start()

        chunk_idx = 0
        try:
            while True:
                if duration_seconds is not None and (time.monotonic() - started) >= duration_seconds:
                    break
                try:
                    audio = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if audio is None:
                    break

                text = self._transcribe_audio_array(audio)
                t0 = chunk_idx * self.config.chunk_seconds
                t1 = t0 + self.config.chunk_seconds
                chunk = TranscriptChunk(text=text, start=t0, end=t1, is_final=True)
                chunk_idx += 1
                if on_chunk:
                    on_chunk(chunk)
                yield chunk
        finally:
            stop_event.set()
            audio_q.put(None)
            producer.join(timeout=2.0)

    def _transcribe_audio_array(self, audio: np.ndarray) -> str:
        """Transcribe a raw float32 mono PCM buffer."""
        if self._backend == "stub" or self._model is None:
            rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
            if rms < 0.01:
                return ""
            return "[STUB] Frammento audio rilevato."

        if self._backend == "riva":
            # Streaming Riva would use streaming_recognize; keep offline path simple.
            return ""

        segments, _ = self._model.transcribe(
            audio,
            language=self.config.language,
            beam_size=1,
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments if seg.text.strip())

    def transcribe_live(
        self,
        duration_seconds: float = 10.0,
        on_chunk: Optional[Callable[[TranscriptChunk], None]] = None,
    ) -> str:
        """Run live transcription for ``duration_seconds`` and return joined text."""
        parts: list[str] = []
        for chunk in self.stream_microphone(duration_seconds=duration_seconds, on_chunk=on_chunk):
            if chunk.text:
                parts.append(chunk.text)
        return " ".join(parts).strip()
