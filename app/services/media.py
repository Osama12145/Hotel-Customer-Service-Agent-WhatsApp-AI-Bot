from __future__ import annotations

import logging

from app.config import get_settings
from app.services.openrouter import OpenRouterService


logger = logging.getLogger(__name__)


class MediaService:
    """Turns audio/image payloads into grounded text before the agent sees them."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.openrouter = OpenRouterService()

    async def transcribe_audio(self, *, base64_audio: str, mime_type: str | None) -> str:
        return await self.transcribe_audio_with_trace(
            base64_audio=base64_audio,
            mime_type=mime_type,
            trace_id=None,
            session_id=None,
            user_id=None,
        )

    async def transcribe_audio_with_trace(
        self,
        *,
        base64_audio: str,
        mime_type: str | None,
        trace_id: str | None,
        session_id: str | None,
        user_id: str | None,
    ) -> str:
        audio_format = self._normalize_audio_format(mime_type)
        response = await self.openrouter.chat(
            model=self.settings.openrouter_audio_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You transcribe customer audio messages for a hotel WhatsApp assistant. "
                        "Return the spoken text only in the same language, without commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "حوّل الرسالة الصوتية إلى نص فقط."},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64_audio,
                                "format": audio_format,
                            },
                        },
                    ],
                },
            ],
            trace_id=trace_id,
            generation_name="audio-transcription",
            session_id=session_id,
            user_id=user_id,
            metadata={"input_kind": "audio", "mime_type": mime_type},
        )
        return response["choices"][0]["message"]["content"]

    async def analyze_image(
        self,
        *,
        base64_image: str,
        mime_type: str | None,
        caption: str | None,
    ) -> str:
        return await self.analyze_image_with_trace(
            base64_image=base64_image,
            mime_type=mime_type,
            caption=caption,
            trace_id=None,
            session_id=None,
            user_id=None,
        )

    async def analyze_image_with_trace(
        self,
        *,
        base64_image: str,
        mime_type: str | None,
        caption: str | None,
        trace_id: str | None,
        session_id: str | None,
        user_id: str | None,
    ) -> str:
        media_type = mime_type or "image/jpeg"
        prompt = caption or "صف محتوى الصورة واستخرج أي معلومات مفيدة لخدمة عميل الفندق."
        response = await self.openrouter.chat(
            model=self.settings.openrouter_vision_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyze customer images for a hotel WhatsApp assistant. "
                        "Extract the useful information from the image in the customer's language."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{base64_image}",
                            },
                        },
                    ],
                },
            ],
            trace_id=trace_id,
            generation_name="image-analysis",
            session_id=session_id,
            user_id=user_id,
            metadata={"input_kind": "image", "mime_type": media_type},
        )
        return response["choices"][0]["message"]["content"]

    @staticmethod
    def _normalize_audio_format(mime_type: str | None) -> str:
        if not mime_type:
            return "ogg"
        if "mpeg" in mime_type or "mp3" in mime_type:
            return "mp3"
        if "wav" in mime_type:
            return "wav"
        if "mp4" in mime_type or "m4a" in mime_type or "aac" in mime_type:
            return "mp4"
        return "ogg"
