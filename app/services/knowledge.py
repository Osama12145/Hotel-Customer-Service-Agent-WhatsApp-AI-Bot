from __future__ import annotations

from pathlib import Path
import re

from app.config import get_settings
from app.knowledge.default_hotel import DEFAULT_HOTEL_KNOWLEDGE


class KnowledgeService:
    """Simple chunk retrieval over markdown knowledge without external vector DB."""

    def __init__(self) -> None:
        settings = get_settings()
        self.knowledge_path = Path(settings.knowledge_file)
        self.knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.knowledge_path.exists():
            self.knowledge_path.write_text(DEFAULT_HOTEL_KNOWLEDGE, encoding="utf-8")

    def _load_text(self) -> str:
        return self.knowledge_path.read_text(encoding="utf-8")

    def retrieve(self, query: str, max_chunks: int = 4) -> str:
        text = self._load_text()
        sections = [section.strip() for section in text.split("\n## ") if section.strip()]
        query_terms = self._terms(query)
        topic_hints = self._topic_hints(query)
        scored_sections: list[tuple[int, str]] = []
        for section in sections:
            lowered = section.lower()
            score = sum(1 for term in query_terms if term in lowered)
            score += sum(weight for hint, weight in topic_hints.items() if hint in lowered)
            if score:
                scored_sections.append((score, section))

        if not scored_sections:
            return text[:2500]

        scored_sections.sort(key=lambda item: item[0], reverse=True)
        return "\n\n".join(section for _, section in scored_sections[:max_chunks])

    @staticmethod
    def _terms(query: str) -> set[str]:
        normalized = re.sub(r"[^\w\u0600-\u06FF]+", " ", query.lower())
        terms = set()
        for term in normalized.split():
            if len(term) <= 2:
                continue
            terms.add(term)
            if term.startswith("ال") and len(term) > 4:
                terms.add(term[2:])
        return terms

    @staticmethod
    def _topic_hints(query: str) -> dict[str, int]:
        lowered = query.lower()
        hints: dict[str, int] = {}
        pricing_words = [
            "سعر",
            "أسعار",
            "كم",
            "تكلفة",
            "ريال",
            "price",
            "rate",
            "cost",
            "harga",
            "fiyat",
            "قیمت",
        ]
        nearby_words = [
            "قريب",
            "مطعم",
            "صيدلية",
            "مواقف",
            "مطار",
            "near",
            "nearby",
            "parking",
            "restaurant",
            "dekat",
            "yakın",
        ]
        if any(word in lowered for word in pricing_words):
            hints["الأسعار"] = 8
            hints["prices"] = 8
        if any(word in lowered for word in nearby_words):
            hints["الخدمات والمناطق القريبة"] = 8
            hints["nearby"] = 8
        return hints
