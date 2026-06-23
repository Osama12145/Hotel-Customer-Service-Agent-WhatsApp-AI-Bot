from __future__ import annotations

import re


class LanguageService:
    """Small deterministic language hint for the supported hotel languages."""

    SUPPORTED_LANGUAGES = {"ar", "en", "ur", "id", "tr"}

    _ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
    _LATIN_RE = re.compile(r"[A-Za-z]")
    _URDU_HINTS = {
        "ہے",
        "ہیں",
        "کیا",
        "آپ",
        "میں",
        "کا",
        "کی",
        "کے",
        "براہ",
        "مہربانی",
    }
    _TURKISH_HINTS = {
        "merhaba",
        "otel",
        "oda",
        "rezervasyon",
        "fiyat",
        "giriş",
        "çıkış",
        "teşekkür",
        "var mı",
    }
    _INDONESIAN_HINTS = {
        "halo",
        "kamar",
        "harga",
        "pesan",
        "reservasi",
        "check in",
        "check-out",
        "terima kasih",
        "berapa",
        "dekat",
    }
    _ENGLISH_HINTS = {
        "hello",
        "room",
        "price",
        "book",
        "booking",
        "reservation",
        "check in",
        "check-out",
        "near",
        "thanks",
    }

    def detect(self, text: str | None) -> str:
        lowered = (text or "").strip().lower()
        if not lowered:
            return "unknown"

        if self._ARABIC_RE.search(lowered):
            if any(hint in lowered for hint in self._URDU_HINTS):
                return "ur"
            return "ar"

        if self._LATIN_RE.search(lowered):
            scores = {
                "tr": self._score(lowered, self._TURKISH_HINTS),
                "id": self._score(lowered, self._INDONESIAN_HINTS),
                "en": self._score(lowered, self._ENGLISH_HINTS),
            }
            best_language, best_score = max(scores.items(), key=lambda item: item[1])
            return best_language if best_score else "en"

        return "unknown"

    @staticmethod
    def _score(text: str, hints: set[str]) -> int:
        return sum(1 for hint in hints if hint in text)
