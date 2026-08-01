"""Deep regression coverage for bugs found in Confu V1.0 QA round 2."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
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
        "IBAN IT60X0542811101000000123456",
        "IBAN IT60 X 05428 11101 000000123456",
        "IBAN IT60X 0542 8111 0100 0000 1234 56",
        "IBAN IT60-X-0542811101000000123456",
        "IT 60 X 0542811101000000123456",
        "bonifico it60x0542811101000000123456 fatto",
    ],
)
def test_iban_formats_round2(tmp_path: Path, text: str) -> None:
    out = _pii(tmp_path).anonymize(text, persist=False)
    assert "[IBAN_" in out.anonymized_text
    assert "0542811101000000123456" not in out.anonymized_text.replace(" ", "").replace("-", "")


@pytest.mark.parametrize(
    "text",
    [
        "tel 3471234567",
        "tel 347-1234567",
        "tel 347.123.4567",
        "tel +39 347 1234567",
        "tel 06 12345678",
    ],
)
def test_phone_formats_round2(tmp_path: Path, text: str) -> None:
    out = _pii(tmp_path).anonymize(text, persist=False)
    assert "[PHONE_" in out.anonymized_text, out.anonymized_text


def test_glued_cf_iban(tmp_path: Path) -> None:
    text = "RSSMRA80A01H501UIT60X0542811101000000123456"
    out = _pii(tmp_path).anonymize(text, persist=False)
    assert "[CF_" in out.anonymized_text
    assert "[IBAN_" in out.anonymized_text
    assert "RSSMRA80A01H501U" not in out.anonymized_text.upper()


def test_asr_split_cf_with_space(tmp_path: Path) -> None:
    """Final joined live transcript often inserts a space mid-CF."""
    text = "Il codice fiscale è RSSMRA80 A01H501U e IBAN IT60X0542811101000000123456"
    out = _pii(tmp_path).anonymize(text, persist=False)
    assert "[CF_" in out.anonymized_text
    assert "RSSMRA80" not in out.anonymized_text
    assert "A01H501U" not in out.anonymized_text
    assert "[IBAN_" in out.anonymized_text


def test_rehydrate_does_not_expand_inside_values(tmp_path: Path) -> None:
    pii = _pii(tmp_path)
    pii.mapping = {"[CF_0]": "contains [CF_1] literal", "[CF_1]": "REALCF"}
    restored = pii.rehydrate("X [CF_0] Y [CF_1] Z")
    assert restored == "X contains [CF_1] literal Y REALCF Z"


def test_directory_audio_rejected(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audiodir"
    audio_dir.mkdir()
    pipe = ConfuV1Pipeline(load_models=False, mapping_path=tmp_path / "m.json")
    with pytest.raises(ValueError, match="not a file"):
        pipe.process_audio_file(audio_dir)


def test_cli_mapping_flag_before_subcommand(tmp_path: Path) -> None:
    """Argparse must not let subparser defaults override --mapping."""
    import wave

    import numpy as np

    wav = tmp_path / "t.wav"
    audio = (np.random.randn(8000) * 0.001 * 32767).astype(np.int16)
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio.tobytes())

    custom_map = tmp_path / "custom" / "map.json"
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "--mapping",
            str(custom_map),
            "file",
            str(wav),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert r.returncode == 0, r.stderr
    assert custom_map.exists()
    assert str(custom_map) in r.stdout


def test_cli_corrupt_mapping_json(tmp_path: Path) -> None:
    corrupt = tmp_path / "bad.json"
    corrupt.write_text("{not json", encoding="utf-8")
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "--mapping",
            str(corrupt),
            "rehydrate",
            "Ciao [CF_0]",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert r.returncode == 2
    assert "ERROR:" in r.stderr
    assert "Traceback" not in r.stderr


def test_cli_negative_duration(tmp_path: Path) -> None:
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "--mapping",
            str(tmp_path / "m.json"),
            "live",
            "--duration",
            "-1",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert r.returncode == 2
    assert "duration" in r.stderr.lower()


def test_concurrent_mapping_writes_valid_json(tmp_path: Path) -> None:
    mapping_path = tmp_path / "race.json"
    pii = PIIFilter(load_model=False, mapping_path=mapping_path)

    def worker(i: int) -> None:
        pii.anonymize(f"email user{i}@example.it CF RSSMRA80A01H501U", persist=True)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data


def test_live_backend_select_still_ok() -> None:
    name = STTEngine._select_audio_backend()
    assert name in {"sounddevice", "pyaudio"}
