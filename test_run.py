#!/usr/bin/env python3
"""
Confu V1.0 — smoke / integration test script.

Verifies:
  1. File audio .wav flow (synthetic tone + stub/real STT + PII masking)
  2. Live microphone capture for ~10 seconds (skipped if no input device)
  3. rehydrate_text round-trip via mapping_dict.json
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

# Ensure project root is on PYTHONPATH when run as script
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from confu_core import ConfuV1Pipeline
from privacy.pii_filter import PIIFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_run")


def _write_silent_wav(path: Path, duration_s: float = 1.0, sr: int = 16000) -> Path:
    """Generate a short silent (near-zero) WAV for file-pipeline smoke tests."""
    n = int(duration_s * sr)
    # Tiny noise so VAD / stub can treat it as present audio
    audio = (np.random.randn(n) * 0.001 * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    return path


def test_pii_anonymize_and_rehydrate() -> None:
    """Unit-level check of Italian PII regex + rehydrate (no heavy models)."""
    logger.info("=== TEST: PII anonymize + rehydrate ===")
    pii = PIIFilter(load_model=False, mapping_path=ROOT / "mapping_dict.json")
    pii.clear_mapping(delete_file=False)

    sample = (
        "Buongiorno, sono Mario Rossi, codice fiscale RSSMRA80A01H501U, "
        "IBAN IT60X0542811101000000123456, telefono +39 347 1234567, "
        "email mario.rossi@example.it."
    )
    result = pii.anonymize(sample, persist=True)

    assert "[" in result.anonymized_text, "Expected placeholders in anonymized text"
    assert "RSSMRA80A01H501U" not in result.anonymized_text, "CF should be masked"
    assert "IT60X0542811101000000123456" not in result.anonymized_text.upper().replace(" ", ""), (
        "IBAN should be masked"
    )
    assert result.mapping, "Mapping must not be empty"

    restored = pii.rehydrate(result.anonymized_text)
    assert "RSSMRA80A01H501U" in restored.upper()
    assert "mario.rossi@example.it" in restored.lower()

    logger.info("Anonymized: %s", result.anonymized_text)
    logger.info("Mapping: %s", json.dumps(result.mapping, ensure_ascii=False))
    logger.info("Rehydrated: %s", restored)
    logger.info("PASS: PII anonymize + rehydrate")


def test_file_pipeline() -> None:
    """End-to-end file flow (works with stub STT if models are missing)."""
    logger.info("=== TEST: process_audio_file (.wav) ===")
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = _write_silent_wav(Path(tmp) / "sample.wav", duration_s=1.5)
        mapping_path = Path(tmp) / "mapping_dict.json"

        pipeline = ConfuV1Pipeline(
            mapping_path=mapping_path,
            load_models=True,  # may fall back to stub/regex
        )
        result = pipeline.process_audio_file(wav_path)

        assert result.source == "file"
        assert isinstance(result.raw_transcript, str)
        assert isinstance(result.anonymized_text, str)
        assert mapping_path.exists() or not result.mapping

        # Stub transcript embeds known PII → expect masking
        if "RSSMRA80A01H501U" in result.raw_transcript.upper():
            assert "RSSMRA80A01H501U" not in result.anonymized_text.upper()
            restored = pipeline.rehydrate_text(result.anonymized_text, result.mapping)
            assert "RSSMRA80A01H501U" in restored.upper()

        logger.info("Raw : %s", result.raw_transcript[:200])
        logger.info("Anon: %s", result.anonymized_text[:200])
        logger.info("PASS: process_audio_file")


def test_live_microphone(duration: float = 10.0) -> None:
    """Live mic test (~10s). Skipped gracefully if no capture device/backend."""
    logger.info("=== TEST: start_live_transcription (%.0fs) ===", duration)

    from stt.engine import STTEngine

    try:
        STTEngine._select_audio_backend()
    except RuntimeError as exc:
        logger.warning("%s — skipping live test", exc)
        return

    # Probe for an input device before starting
    has_device = False
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        has_device = any(d.get("max_input_channels", 0) > 0 for d in devices)
    except Exception:
        try:
            import pyaudio

            pa = pyaudio.PyAudio()
            has_device = any(
                pa.get_device_info_by_index(i).get("maxInputChannels", 0) > 0
                for i in range(pa.get_device_count())
            )
            pa.terminate()
        except Exception as exc:
            logger.warning("No audio backend available (%s) — skipping live test", exc)
            return

    if not has_device:
        logger.warning("No microphone input device found — skipping live test")
        return

    with tempfile.TemporaryDirectory() as tmp:
        mapping_path = Path(tmp) / "mapping_dict.json"
        pipeline = ConfuV1Pipeline(mapping_path=mapping_path, load_models=True)

        def on_partial(raw: str, anon: str) -> None:
            logger.info("chunk raw=%r anon=%r", raw, anon)

        try:
            result = pipeline.start_live_transcription(
                duration_seconds=duration,
                on_partial=on_partial,
            )
        except RuntimeError as exc:
            logger.warning("Live capture failed (%s) — skipping", exc)
            return

        assert result.source == "live"
        logger.info(
            "Live transcript (%d chars): %s",
            len(result.raw_transcript),
            result.raw_transcript[:200],
        )
        logger.info("PASS: start_live_transcription")


def main() -> int:
    logger.info("Confu V1.0 — test_run starting")
    test_pii_anonymize_and_rehydrate()
    test_file_pipeline()

    # Live test: 10 seconds as requested in the handover prompt
    skip_live = "--skip-live" in sys.argv
    if skip_live:
        logger.info("Skipping live mic test (--skip-live)")
    else:
        test_live_microphone(duration=10.0)

    logger.info("All requested tests finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
