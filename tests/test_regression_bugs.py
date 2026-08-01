"""Regression tests for PII masking bugs found during Confu V1.0 QA."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from confu_core import ConfuV1Pipeline
from privacy.pii_filter import PIIFilter
from stt.engine import STTEngine


def _pii(tmp_path: Path) -> PIIFilter:
    return PIIFilter(load_model=False, mapping_path=tmp_path / "mapping_dict.json")


@pytest.mark.parametrize(
    "text",
    [
        "IBAN IT60X0542811101000000123456 fine",
        "IBAN IT60 X 05428 11101 000000123456 fine",
        "IBAN IT60X 0542 8111 0100 0000 1234 56 fine",
        "iban it60x0542811101000000123456 fine",
    ],
)
def test_iban_formats_are_fully_masked(tmp_path: Path, text: str) -> None:
    out = _pii(tmp_path).anonymize(text, persist=False)
    compact_in = "IT60X0542811101000000123456"
    compact_out = out.anonymized_text.upper().replace(" ", "")
    assert compact_in not in compact_out
    assert "[IBAN_" in out.anonymized_text
    # Must not leave a phone-masked tail that leaks most of the IBAN
    assert "05428" not in out.anonymized_text
    assert "11101" not in out.anonymized_text


def test_phone_does_not_eat_mid_number(tmp_path: Path) -> None:
    out = _pii(tmp_path).anonymize("importo 12345678901 euro", persist=False)
    assert out.anonymized_text == "importo 12345678901 euro"
    assert "[PHONE_" not in out.anonymized_text


def test_spaced_iban_not_partially_phoned(tmp_path: Path) -> None:
    text = "Bonifico IT60 X 05428 11101 000000123456 grazie"
    out = _pii(tmp_path).anonymize(text, persist=False)
    assert "[IBAN_" in out.anonymized_text
    assert "[PHONE_" not in out.anonymized_text
    assert "05428" not in out.anonymized_text


def test_real_phones_still_masked(tmp_path: Path) -> None:
    cases = [
        "tel 3471234567",
        "tel 347 1234567",
        "tel +39 347 1234567",
        "tel 06 12345678",
        "tel 0212345678",
    ]
    for text in cases:
        out = _pii(tmp_path).anonymize(text, persist=False)
        assert "[PHONE_" in out.anonymized_text, text
        digits = "".join(ch for ch in text if ch.isdigit())
        # national significant digits should not remain contiguous in output
        assert digits[-7:] not in out.anonymized_text.replace(" ", "")


def test_piva_contextual_only(tmp_path: Path) -> None:
    out = _pii(tmp_path).anonymize(
        "Partita IVA 12345678901 e importo 12345678901 euro", persist=False
    )
    assert out.anonymized_text.count("[PIVA_") == 1
    assert "importo 12345678901 euro" in out.anonymized_text


def test_save_mapping_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "mapping_dict.json"
    pii = PIIFilter(load_model=False, mapping_path=nested)
    pii.anonymize("CF RSSMRA80A01H501U", persist=True)
    assert nested.exists()
    data = json.loads(nested.read_text(encoding="utf-8"))
    assert data["[CF_0]"].upper() == "RSSMRA80A01H501U"


def test_iban_reuse_ignores_spacing(tmp_path: Path) -> None:
    pii = _pii(tmp_path)
    out = pii.anonymize(
        "IT60X0542811101000000123456 e IT60 X 05428 11101 000000123456",
        persist=False,
    )
    assert out.anonymized_text.count("[IBAN_0]") == 2
    assert sum(1 for k in out.mapping if k.startswith("[IBAN_")) == 1


def test_live_backend_missing_raises() -> None:
    # In this environment sounddevice/pyaudio may be missing — expect clear error
    try:
        STTEngine._select_audio_backend()
    except RuntimeError as exc:
        assert "sounddevice" in str(exc) or "pyaudio" in str(exc)
        return
    # If a backend is installed, selecting it must return a known name
    assert STTEngine._select_audio_backend() in {"sounddevice", "pyaudio"}


def test_cli_missing_file_exit_code(tmp_path: Path) -> None:
    import subprocess

    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "--mapping",
            str(tmp_path / "m.json"),
            "file",
            str(tmp_path / "missing.wav"),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert r.returncode == 2
    assert "not found" in r.stderr.lower() or "ERROR" in r.stderr


def test_cli_verbose_after_subcommand(tmp_path: Path) -> None:
    import subprocess

    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "file",
            "-v",
            "--mapping",
            str(tmp_path / "m.json"),
            str(tmp_path / "missing.wav"),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    # Must parse OK (not argparse error) and then fail on missing file
    assert r.returncode == 2
    assert "unrecognized arguments" not in r.stderr
    assert "ERROR:" in r.stderr


def test_cli_live_no_device_exits_clean(tmp_path: Path) -> None:
    import subprocess

    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "--mapping",
            str(tmp_path / "m.json"),
            "live",
            "--duration",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    # No mic in CI → RuntimeError surfaced as exit 2 (not traceback crash)
    assert r.returncode == 2
    assert "ERROR:" in r.stderr
    assert "Traceback" not in r.stderr


def test_pipeline_file_roundtrip(tmp_path: Path) -> None:
    import wave

    import numpy as np

    wav = tmp_path / "sample.wav"
    audio = (np.random.randn(8000) * 0.001 * 32767).astype(np.int16)
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio.tobytes())

    pipeline = ConfuV1Pipeline(
        mapping_path=tmp_path / "mapping_dict.json",
        load_models=True,
    )
    result = pipeline.process_audio_file(wav)
    assert result.source == "file"
    if "RSSMRA80A01H501U" in result.raw_transcript.upper():
        assert "RSSMRA80A01H501U" not in result.anonymized_text.upper()
        restored = pipeline.rehydrate_text(result.anonymized_text, result.mapping)
        assert "RSSMRA80A01H501U" in restored.upper()
