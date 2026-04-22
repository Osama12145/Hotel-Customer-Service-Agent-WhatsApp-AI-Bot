from __future__ import annotations

from pathlib import Path

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
        query_terms = {term.strip(".,!?").lower() for term in query.split() if len(term) > 2}
        scored_sections: list[tuple[int, str]] = []
        for section in sections:
            lowered = section.lower()
            score = sum(1 for term in query_terms if term in lowered)
            if score:
                scored_sections.append((score, section))

        if not scored_sections:
            return text[:2500]

        scored_sections.sort(key=lambda item: item[0], reverse=True)
        return "\n\n".join(section for _, section in scored_sections[:max_chunks])
