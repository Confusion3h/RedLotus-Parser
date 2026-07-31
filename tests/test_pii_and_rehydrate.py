"""Lightweight unit tests (no GPU / no microphone required)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from confu_core import ConfuV1Pipeline
from privacy.pii_filter import PIIFilter


def test_regex_pii_masks_italian_identifiers(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping_dict.json"
    pii = PIIFilter(load_model=False, mapping_path=mapping)
    text = (
        "Cliente: Giulia Bianchi CF BNCGLL90C41F205Z "
        "IBAN IT60X0542811101000000123456 email giulia@acme.it tel 3331234567"
    )
    out = pii.anonymize(text, persist=True)
    assert "BNCGLL90C41F205Z" not in out.anonymized_text.upper()
    assert "IT60X0542811101000000123456" not in out.anonymized_text.upper().replace(" ", "")
    assert "giulia@acme.it" not in out.anonymized_text.lower()
    assert mapping.exists()
    restored = pii.rehydrate(out.anonymized_text)
    assert "BNCGLL90C41F205Z" in restored.upper()
    assert "giulia@acme.it" in restored.lower()


def test_pipeline_rehydrate_roundtrip(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping_dict.json"
    pipeline = ConfuV1Pipeline(mapping_path=mapping_path, load_models=False)
    # Seed mapping manually
    pipeline.pii.mapping = {"[FULLNAME_0]": "Luca Verdi", "[CF_0]": "VRDLCU85T12A662P"}
    pipeline.save_mapping()
    anon = "Gentile [FULLNAME_0], CF [CF_0]."
    restored = pipeline.rehydrate_text(anon)
    assert restored == "Gentile Luca Verdi, CF VRDLCU85T12A662P."
    loaded = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert loaded["[FULLNAME_0]"] == "Luca Verdi"
