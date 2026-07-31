"""
ConfuV1Pipeline — unified Compound AI System for local STT + Italian PII masking.

Flow:
  audio (file | live mic) → local STT → local PII filter → anonymized text + mapping
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from privacy.pii_filter import AnonymizationResult, PIIFilter
from stt.engine import STTConfig, STTEngine, TranscriptChunk

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Final output of a Confu processing run."""

    raw_transcript: str
    anonymized_text: str
    mapping: dict[str, str] = field(default_factory=dict)
    entities: list[dict[str, Any]] = field(default_factory=list)
    source: str = "file"  # file | live


class ConfuV1Pipeline:
    """
    Compound AI System: local batch & real-time STT + local Italian PII anonymizer.

    Methods
    -------
    process_audio_file(audio_path)
        Transcribe a recorded file, detect PII, return censored text + mapping.
    start_live_transcription(duration_seconds=None, on_partial=None)
        Capture microphone audio, stream-transcribe in chunks, anonymize on the fly.
    rehydrate_text(anonymized_text, mapping_dict=None)
        Restore original PII values from placeholders using the reserved map.
    """

    def __init__(
        self,
        stt_model_size: str = "medium",
        stt_device: str = "auto",
        stt_compute_type: str = "auto",
        pii_model_id: str = "rizzoaiacademy/rizzo-pii-0.3B",
        mapping_path: str | Path = "mapping_dict.json",
        language: str = "it",
        riva_uri: Optional[str] = None,
        load_models: bool = True,
    ) -> None:
        logger.info("Initializing ConfuV1Pipeline …")

        stt_config = STTConfig(
            model_size=stt_model_size,
            device=stt_device,
            compute_type=stt_compute_type,
            language=language,
            riva_uri=riva_uri,
        )
        self.stt = STTEngine(stt_config, load_model=load_models)
        self.pii = PIIFilter(
            model_id=pii_model_id,
            mapping_path=mapping_path,
            load_model=load_models,
        )
        logger.info(
            "ConfuV1Pipeline ready (stt=%s, pii=%s)",
            self.stt._backend,
            self.pii._backend,
        )

    # ----------------------------------------------------------- batch
    def process_audio_file(self, audio_path: str | Path) -> PipelineResult:
        """
        Batch mode: file → STT → PII mask → anonymized text + mapping.
        """
        path = Path(audio_path)
        logger.info("Processing audio file: %s", path)

        raw = self.stt.transcribe_file(path)
        anon: AnonymizationResult = self.pii.anonymize(raw, persist=True)

        result = PipelineResult(
            raw_transcript=raw,
            anonymized_text=anon.anonymized_text,
            mapping=anon.mapping,
            entities=anon.entities,
            source="file",
        )
        logger.info(
            "File done — raw=%d chars, anon=%d chars, entities=%d",
            len(result.raw_transcript),
            len(result.anonymized_text),
            len(result.entities),
        )
        return result

    # ----------------------------------------------------------- live
    def start_live_transcription(
        self,
        duration_seconds: Optional[float] = 10.0,
        on_partial: Optional[Callable[[str, str], None]] = None,
    ) -> PipelineResult:
        """
        Live / streaming mode: microphone → chunked STT → incremental PII mask.

        Parameters
        ----------
        duration_seconds:
            Stop after N seconds. ``None`` keeps going until KeyboardInterrupt
            (caller must handle that). Default 10s for safe demos.
        on_partial:
            Optional callback ``(raw_chunk, anonymized_chunk)`` fired per chunk.
        """
        logger.info(
            "Starting live transcription (duration=%s) …",
            duration_seconds if duration_seconds is not None else "∞",
        )

        raw_parts: list[str] = []
        anon_parts: list[str] = []

        def _handle(chunk: TranscriptChunk) -> None:
            if not chunk.text:
                return
            raw_parts.append(chunk.text)
            anon = self.pii.anonymize(chunk.text, persist=False)
            anon_parts.append(anon.anonymized_text)
            if on_partial:
                on_partial(chunk.text, anon.anonymized_text)

        try:
            for _ in self.stt.stream_microphone(
                duration_seconds=duration_seconds,
                on_chunk=_handle,
            ):
                pass
        except KeyboardInterrupt:
            logger.info("Live transcription interrupted by user")
        except RuntimeError:
            # Surface mic/backend errors to the caller (CLI handles messaging)
            raise

        # Persist mapping once at the end
        self.pii.save_mapping()

        raw = " ".join(raw_parts).strip()
        anonymized = " ".join(anon_parts).strip()

        # Final full-pass anonymization for consistency across chunk boundaries
        if raw:
            final = self.pii.anonymize(raw, persist=True)
            anonymized = final.anonymized_text
            entities = final.entities
            mapping = final.mapping
        else:
            entities = []
            mapping = dict(self.pii.mapping)

        result = PipelineResult(
            raw_transcript=raw,
            anonymized_text=anonymized,
            mapping=mapping,
            entities=entities,
            source="live",
        )
        logger.info(
            "Live done — raw=%d chars, anon=%d chars, entities=%d",
            len(result.raw_transcript),
            len(result.anonymized_text),
            len(result.entities),
        )
        return result

    # -------------------------------------------------------- rehydrate
    def rehydrate_text(
        self,
        anonymized_text: str,
        mapping_dict: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Utility: convert censored text back to real values using the reserved map.
        """
        return self.pii.rehydrate(anonymized_text, mapping=mapping_dict)

    # -------------------------------------------------------- helpers
    def get_mapping(self) -> dict[str, str]:
        return dict(self.pii.mapping)

    def save_mapping(self, path: Optional[str | Path] = None) -> Path:
        return self.pii.save_mapping(path)
