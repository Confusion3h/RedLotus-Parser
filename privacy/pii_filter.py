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

# Codice Fiscale: 16 alphanumeric chars with month letter set
_CF_RE = re.compile(
    r"\b([A-Z]{6}\d{2}[A-EHLMPRST]\d{2}[A-Z]\d{3}[A-Z])\b",
    re.IGNORECASE,
)

# Italian IBAN: IT + 2 check digits + 1 CIN letter + 22 digits (spaces allowed anywhere)
# Total 27 alphanumeric characters. Example: IT60X0542811101000000123456
_IBAN_RE = re.compile(
    r"\b(IT(?:\s*\d){2}\s*[A-Z](?:\s*\d){22})\b",
    re.IGNORECASE,
)

# Partita IVA: optional IT prefix + exactly 11 digits
_PIVA_RE = re.compile(r"\b(IT\s*\d{11}|\d{11})\b", re.IGNORECASE)

# Italian phones: optional +39, mobile 3XX… or landline 0… ; no mid-digit matches
_PHONE_RE = re.compile(
    r"(?<!\d)(\+39[\s\-]?)?(?:0\d{1,3}[\s\-]?\d{6,8}|3\d{2}[\s\-]?\d{6,7})(?!\d)"
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


def _digits_only(value: str) -> str:
    return re.sub(r"\D", "", value)


def _is_plausible_italian_phone(value: str) -> bool:
    """Reject mid-number / account-tail matches that only look phone-like."""
    digits = _digits_only(value)
    if digits.startswith("39") and len(digits) >= 11:
        digits = digits[2:]
    # Italian national numbers are typically 9–11 digits
    if not (9 <= len(digits) <= 11):
        return False
    if digits.startswith("3"):
        return len(digits) in (9, 10)
    if digits.startswith("0"):
        return True
    return False


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
                    for key in self.mapping:
                        body = key.strip("[]")
                        if "_" not in body:
                            continue
                        prefix, idx_part = body.rsplit("_", 1)
                        if idx_part.isdigit():
                            self._counters[prefix] = max(
                                self._counters.get(prefix, 0), int(idx_part) + 1
                            )
                    logger.info(
                        "Loaded %d mapping entries from %s",
                        len(self.mapping),
                        self.mapping_path,
                    )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read mapping file: %s", exc)

    def save_mapping(self, path: Optional[str | Path] = None) -> Path:
        """Persist the placeholder ↔ real-value map to a local JSON file."""
        out = Path(path) if path else self.mapping_path
        out.parent.mkdir(parents=True, exist_ok=True)
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
        # Normalize IBAN/CF comparisons by stripping spaces
        norm_value = re.sub(r"\s+", "", value).lower()
        for ph, real in self.mapping.items():
            if not ph.startswith(f"[{prefix}_"):
                continue
            if re.sub(r"\s+", "", real).lower() == norm_value:
                return ph
        ph = self._next_placeholder(prefix)
        self.mapping[ph] = value
        return ph

    @staticmethod
    def _normalize_label(label: str) -> str:
        clean = (
            label.upper()
            .replace("B-", "")
            .replace("I-", "")
            .replace("L-", "")
            .replace("U-", "")
        )
        return LABEL_TO_PREFIX.get(
            clean, clean if clean in LABEL_TO_PREFIX.values() else "PII"
        )

    # ----------------------------------------------------------- detect
    def detect_entities(self, text: str) -> list[dict[str, Any]]:
        """Return list of {start, end, label, text, score} spans."""
        entities: list[dict[str, Any]] = []

        if self._pipeline is not None:
            try:
                raw = self._pipeline(text)
                for ent in raw:
                    start = int(ent["start"])
                    end = int(ent["end"])
                    # Prefer original slice so offsets stay consistent after strip
                    word = text[start:end]
                    entities.append(
                        {
                            "start": start,
                            "end": end,
                            "label": self._normalize_label(
                                str(ent.get("entity_group") or ent.get("entity") or "PII")
                            ),
                            "text": word,
                            "score": float(ent.get("score", 1.0)),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("NER pipeline failed (%s); continuing with regex.", exc)

        entities.extend(self._regex_entities(text))
        return self._dedupe_spans(entities)

    def _regex_entities(self, text: str) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []

        # Priority order: IBAN before phone/PIVA so account tails are not mislabeled
        patterns: list[tuple[re.Pattern[str], str]] = [
            (_IBAN_RE, "IBAN"),
            (_CF_RE, "CF"),
            (_EMAIL_RE, "EMAIL"),
            (_PHONE_RE, "PHONE"),
            (_PIVA_RE, "PIVA"),
        ]
        for pattern, label in patterns:
            for m in pattern.finditer(text):
                value = m.group(0)
                if label == "PHONE" and not _is_plausible_italian_phone(value):
                    continue
                if label == "PIVA":
                    compact = re.sub(r"\s+", "", value).upper()
                    digits = _digits_only(value)
                    if compact.startswith("IT"):
                        pass
                    elif len(digits) == 11:
                        ctx = text[max(0, m.start() - 24) : m.start()].lower()
                        if not any(
                            k in ctx
                            for k in ("p.iva", "piva", "partita", "partita iva", "iva")
                        ):
                            continue
                    else:
                        continue
                if label == "IBAN":
                    compact = re.sub(r"\s+", "", value).upper()
                    if len(compact) != 27 or not compact.startswith("IT"):
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
        """Prefer longer / higher-score spans; IBAN beats PHONE/PIVA on ties."""
        if not entities:
            return []

        priority = {"IBAN": 3, "CF": 3, "EMAIL": 2, "PIVA": 2, "PHONE": 1, "FULLNAME": 2}

        def sort_key(e: dict[str, Any]) -> tuple:
            return (
                e["start"],
                -(e["end"] - e["start"]),
                -priority.get(e.get("label", ""), 0),
                -e.get("score", 0),
            )

        sorted_ents = sorted(entities, key=sort_key)
        kept: list[dict[str, Any]] = []
        last_end = -1
        for ent in sorted_ents:
            if ent["start"] < last_end:
                continue
            if not str(ent.get("text", "")).strip():
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
    def rehydrate(
        self, anonymized_text: str, mapping: Optional[dict[str, str]] = None
    ) -> str:
        """Replace placeholders with original values using the mapping dict."""
        mapping = mapping if mapping is not None else self.mapping
        result = anonymized_text
        for placeholder, real in sorted(
            mapping.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            result = result.replace(placeholder, real)
        return result
