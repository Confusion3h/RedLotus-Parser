"""
Italian PII Filter — local anonymization via rizzoaiacademy/rizzo-pii-0.3B.

Detects Italian PII (names, CF, IBAN, addresses, Partita IVA, phones) and
replaces them with unique placeholders such as [FULLNAME_0], [CF_1], [IBAN_2].
A temporary mapping dict is kept in RAM and optionally persisted to JSON.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "rizzoaiacademy/rizzo-pii-0.3B"
DEFAULT_MAPPING_PATH = Path("mapping_dict.json")

# Regex fallbacks for high-precision Italian identifiers (always applied)
_CF_RE = re.compile(
    r"\b([A-Z]{6}\d{2}[A-EHLMPRST]\d{2}[A-Z]\d{3}[A-Z])\b",
    re.IGNORECASE,
)
_IBAN_RE = re.compile(
    r"\b(IT\d{2}[A-Z]\d{10}\d{12}|IT\d{2}\s?[A-Z]\s?(?:\d{4}\s?){5}\d{3})\b",
    re.IGNORECASE,
)
_PIVA_RE = re.compile(r"\b((?:IT)?\d{11})\b", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"(?:\+39\s?)?(?:0\d{1,3}[\s\-]?\d{6,8}|3\d{2}[\s\-]?\d{6,7})\b"
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# Map model label / regex kind → placeholder prefix
LABEL_TO_PREFIX = {
    "PER": "FULLNAME",
    "PERSON": "FULLNAME",
    "NOME": "FULLNAME",
    "FULLNAME": "FULLNAME",
    "CF": "CF",
    "CODICE_FISCALE": "CF",
    "FISCAL_CODE": "CF",
    "IBAN": "IBAN",
    "LOC": "ADDRESS",
    "ADDRESS": "ADDRESS",
    "INDIRIZZO": "ADDRESS",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "PIVA": "PIVA",
    "PARTITA_IVA": "PIVA",
    "VAT": "PIVA",
    "PHONE": "PHONE",
    "TELEFONO": "PHONE",
    "EMAIL": "EMAIL",
    "MAIL": "EMAIL",
}


@dataclass
class AnonymizationResult:
    """Output of a single anonymization pass."""

    anonymized_text: str
    mapping: dict[str, str] = field(default_factory=dict)
    entities: list[dict[str, Any]] = field(default_factory=list)


class PIIFilter:
    """
    Local Italian PII detector & masker.

    Prefer the HuggingFace token-classification model
    ``rizzoaiacademy/rizzo-pii-0.3B``; if unavailable, fall back to regex-only.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        mapping_path: str | Path = DEFAULT_MAPPING_PATH,
        device: Optional[int | str] = None,
        load_model: bool = True,
    ) -> None:
        self.model_id = model_id
        self.mapping_path = Path(mapping_path)
        self.mapping: dict[str, str] = {}
        self._counters: dict[str, int] = {}
        self._pipeline = None
        self._backend = "regex"

        if load_model:
            self._load_ner(device)

        self._load_mapping_from_disk()

    # ---------------------------------------------------------------- load
    def _load_ner(self, device: Optional[int | str]) -> None:
        try:
            from transformers import pipeline

            pipe_kwargs: dict[str, Any] = {
                "task": "token-classification",
                "model": self.model_id,
                "aggregation_strategy": "simple",
            }
            if device is not None:
                pipe_kwargs["device"] = device

            logger.info("Loading PII model %s …", self.model_id)
            self._pipeline = pipeline(**pipe_kwargs)
            self._backend = "rizzo-pii"
            logger.info("PII model ready (backend=%s)", self._backend)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load %s (%s). Using regex-only Italian PII fallback.",
                self.model_id,
                exc,
            )
            self._pipeline = None
            self._backend = "regex"

    def _load_mapping_from_disk(self) -> None:
        if self.mapping_path.exists():
            try:
                data = json.loads(self.mapping_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.mapping.update({str(k): str(v) for k, v in data.items()})
                    # rebuild counters from existing keys
                    for key in self.mapping:
                        prefix = key.strip("[]").rsplit("_", 1)[0]
                        idx_part = key.strip("[]").rsplit("_", 1)[-1]
                        if idx_part.isdigit():
                            self._counters[prefix] = max(
                                self._counters.get(prefix, 0), int(idx_part) + 1
                            )
                    logger.info("Loaded %d mapping entries from %s", len(self.mapping), self.mapping_path)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read mapping file: %s", exc)

    def save_mapping(self, path: Optional[str | Path] = None) -> Path:
        """Persist the placeholder ↔ real-value map to a local JSON file."""
        out = Path(path) if path else self.mapping_path
        out.write_text(
            json.dumps(self.mapping, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved mapping (%d entries) → %s", len(self.mapping), out)
        return out

    def clear_mapping(self, delete_file: bool = False) -> None:
        self.mapping.clear()
        self._counters.clear()
        if delete_file and self.mapping_path.exists():
            self.mapping_path.unlink()

    # ----------------------------------------------------------- helpers
    def _next_placeholder(self, prefix: str) -> str:
        idx = self._counters.get(prefix, 0)
        self._counters[prefix] = idx + 1
        return f"[{prefix}_{idx}]"

    def _reuse_or_create(self, prefix: str, value: str) -> str:
        """Reuse an existing placeholder for the same value (case-insensitive)."""
        for ph, real in self.mapping.items():
            if real.lower() == value.lower() and ph.startswith(f"[{prefix}_"):
                return ph
        ph = self._next_placeholder(prefix)
        self.mapping[ph] = value
        return ph

    @staticmethod
    def _normalize_label(label: str) -> str:
        clean = label.upper().replace("B-", "").replace("I-", "").replace("L-", "").replace("U-", "")
        return LABEL_TO_PREFIX.get(clean, clean if clean in LABEL_TO_PREFIX.values() else "PII")

    # ----------------------------------------------------------- detect
    def detect_entities(self, text: str) -> list[dict[str, Any]]:
        """Return list of {start, end, label, text, score} spans."""
        entities: list[dict[str, Any]] = []

        if self._pipeline is not None:
            try:
                raw = self._pipeline(text)
                for ent in raw:
                    word = ent.get("word") or text[ent["start"] : ent["end"]]
                    entities.append(
                        {
                            "start": int(ent["start"]),
                            "end": int(ent["end"]),
                            "label": self._normalize_label(str(ent.get("entity_group") or ent.get("entity") or "PII")),
                            "text": word.strip(),
                            "score": float(ent.get("score", 1.0)),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("NER pipeline failed (%s); continuing with regex.", exc)

        # Always enrich with deterministic Italian identifier regexes
        entities.extend(self._regex_entities(text))
        return self._dedupe_spans(entities)

    def _regex_entities(self, text: str) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        patterns = [
            (_CF_RE, "CF"),
            (_IBAN_RE, "IBAN"),
            (_EMAIL_RE, "EMAIL"),
            (_PHONE_RE, "PHONE"),
            (_PIVA_RE, "PIVA"),
        ]
        for pattern, label in patterns:
            for m in pattern.finditer(text):
                # Skip bare 11-digit numbers that look like CF fragments already matched
                value = m.group(0)
                if label == "PIVA" and value.upper().startswith("IT") is False and len(value) == 11:
                    # Avoid matching random long numbers unless prefixed IT or clearly VAT context
                    ctx = text[max(0, m.start() - 20) : m.start()].lower()
                    if "p.iva" not in ctx and "partita" not in ctx and "iva" not in ctx:
                        continue
                found.append(
                    {
                        "start": m.start(),
                        "end": m.end(),
                        "label": label,
                        "text": value,
                        "score": 1.0,
                    }
                )
        return found

    @staticmethod
    def _dedupe_spans(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prefer higher-score / longer spans when overlapping."""
        if not entities:
            return []
        sorted_ents = sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"]), -e.get("score", 0)))
        kept: list[dict[str, Any]] = []
        last_end = -1
        for ent in sorted_ents:
            if ent["start"] < last_end:
                continue
            if not ent["text"]:
                continue
            kept.append(ent)
            last_end = ent["end"]
        return kept

    # -------------------------------------------------------- anonymize
    def anonymize(self, text: str, persist: bool = True) -> AnonymizationResult:
        """
        Detect PII and replace with placeholders. Updates ``self.mapping``.

        Returns anonymized text + mapping snapshot + entity list.
        """
        if not text:
            return AnonymizationResult(anonymized_text="", mapping=dict(self.mapping))

        entities = self.detect_entities(text)
        # Replace from the end so offsets stay valid
        anonymized = text
        for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
            prefix = ent["label"]
            placeholder = self._reuse_or_create(prefix, ent["text"])
            anonymized = anonymized[: ent["start"]] + placeholder + anonymized[ent["end"] :]

        if persist:
            self.save_mapping()

        return AnonymizationResult(
            anonymized_text=anonymized,
            mapping=dict(self.mapping),
            entities=entities,
        )

    # -------------------------------------------------------- rehydrate
    def rehydrate(self, anonymized_text: str, mapping: Optional[dict[str, str]] = None) -> str:
        """Replace placeholders with original values using the mapping dict."""
        mapping = mapping if mapping is not None else self.mapping
        result = anonymized_text
        # Longer keys first to avoid partial collisions
        for placeholder, real in sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True):
            result = result.replace(placeholder, real)
        return result
