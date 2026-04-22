from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from app.config import get_settings
from app.services.langfuse_service import LangfuseService


logger = logging.getLogger(__name__)


class OpenRouterService:
    """OpenRouter client used for text, vision, and audio reasoning."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.langfuse = LangfuseService()

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.2,
        response_format: dict[str, Any] | None = None,
        trace_id: str | None = None,
        generation_name: str = "openrouter-chat",
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        url = f"{self.settings.openrouter_base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

        trace_metadata = {
            "session_id": session_id,
            "user_id": user_id,
        }
        if metadata:
            trace_metadata.update(metadata)
        self.langfuse.log_generation(
            trace_id=trace_id,
            name=generation_name,
            model=model,
            input_payload=payload,
            output_payload=result,
            metadata=trace_metadata,
        )
        self.langfuse.flush()
        return result

    async def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        trace_id: str | None = None,
        generation_name: str = "complete-text",
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        response = await self.chat(
            model=self.settings.openrouter_text_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            trace_id=trace_id,
            generation_name=generation_name,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
        )
        return response["choices"][0]["message"]["content"]

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        trace_id: str | None = None,
        generation_name: str = "complete-json",
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        text = await self.complete_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            trace_id=trace_id,
            generation_name=generation_name,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
        )
        return self.parse_json_response(text)

    @staticmethod
    def parse_json_response(text: str) -> dict[str, Any]:
        cleaned = (text or "").strip()
        if not cleaned:
            raise ValueError("The model returned an empty response.")

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Common model behavior is to wrap JSON in a fenced markdown block.
        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
        if fenced_match:
            return json.loads(fenced_match.group(1))

        # If the model added explanation around JSON, extract the first JSON object.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            return json.loads(candidate)

        logger.warning("Failed to parse JSON response from model: %s", cleaned)
        raise ValueError("The model returned invalid JSON.")
