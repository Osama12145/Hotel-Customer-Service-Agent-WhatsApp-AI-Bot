from __future__ import annotations

import logging

import httpx

from app.config import get_settings


logger = logging.getLogger(__name__)


class EvolutionService:
    """Small client around EvolutionAPI endpoints we need in code."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def send_text(self, instance: str, remote_jid: str, text: str) -> None:
        headers = {"apikey": self.settings.evolution_api_key}
        payload = {"number": remote_jid, "text": text}
        url = f"{self.settings.evolution_base_url}/message/sendText/{instance}"
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        logger.info("Sent WhatsApp reply to %s", remote_jid)

    async def get_media_base64(self, instance: str, message_id: str) -> dict:
        headers = {"apikey": self.settings.evolution_api_key}
        payload = {"message": {"key": {"id": message_id}}, "convertToMp4": True}
        url = f"{self.settings.evolution_base_url}/chat/getBase64FromMediaMessage/{instance}"
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
