from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.agent.graph import HotelAgent
from app.config import get_settings
from app.logging_utils import configure_logging
from app.models import HotelAgentState
from app.services.evolution import EvolutionService
from app.services.langfuse_service import LangfuseService
from app.services.media import MediaService
from app.services.payload_parser import PayloadParser
from app.services.storage import StorageService


configure_logging()
settings = get_settings()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)
storage = StorageService()
evolution = EvolutionService()
media = MediaService()
agent = HotelAgent()
langfuse = LangfuseService()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _normalize_base64_media(
    base64_value: str | None,
    mime_type: str | None,
) -> tuple[str | None, str | None]:
    if not base64_value:
        return None, mime_type

    if base64_value.startswith("data:"):
        header, separator, data = base64_value.partition(",")
        if separator:
            header_mime = header.removeprefix("data:").split(";", 1)[0]
            return data, header_mime or mime_type

    return base64_value, mime_type


def _extract_base64_media(
    media_payload: dict[str, Any],
    fallback_mime_type: str | None,
) -> tuple[str | None, str | None]:
    containers: list[dict[str, Any]] = []
    data = media_payload.get("data")
    if isinstance(data, dict):
        containers.append(data)
    containers.append(media_payload)

    for container in containers:
        base64_value = (
            container.get("base64")
            or container.get("media")
            or container.get("mediaBase64")
        )
        if not base64_value:
            continue
        mime_type = (
            container.get("mimetype")
            or container.get("mimeType")
            or container.get("mediaType")
            or fallback_mime_type
        )
        return _normalize_base64_media(base64_value, mime_type)

    return None, fallback_mime_type


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        parsed = PayloadParser.parse(payload)
        logger.info(
            "Incoming WhatsApp message | session=%s | type=%s | id=%s",
            parsed.remote_jid,
            parsed.message_type,
            parsed.message_id,
        )

        if not parsed.message_id or not parsed.remote_jid:
            logger.info("Ignoring webhook payload without a message id or remote JID")
            return JSONResponse({"status": "ignored_non_message_event"})

        if parsed.from_me:
            logger.info("Ignoring outgoing/self message %s", parsed.message_id)
            return JSONResponse({"status": "ignored_from_me"})

        trace_id, _ = langfuse.create_trace(
            name="whatsapp-webhook",
            user_id=parsed.remote_jid,
            session_id=parsed.remote_jid,
            input_payload={
                "message_id": parsed.message_id,
                "message_type": parsed.message_type,
                "instance": parsed.instance,
            },
            metadata={"customer_name": parsed.push_name},
        )

        if storage.has_processed_message(parsed.message_id):
            return JSONResponse({"status": "duplicate_ignored"})

        incoming_text = await _resolve_incoming_text(parsed, trace_id=trace_id)
        if not incoming_text.strip():
            storage.mark_message_processed(parsed.message_id, parsed.remote_jid)
            return JSONResponse({"status": "ignored_empty"})

        storage.append_message(
            session_id=parsed.remote_jid,
            role="user",
            content=incoming_text,
            message_type=parsed.message_type,
        )

        state = HotelAgentState(
            session_id=parsed.remote_jid,
            incoming_text=incoming_text,
            customer_name=parsed.push_name,
            original_message_type=parsed.message_type,
            message_id=parsed.message_id,
            raw_payload=parsed.raw_payload,
            chat_history=storage.get_recent_history(parsed.remote_jid),
            trace_id=trace_id,
        )

        result = await agent.run(state)

        await evolution.send_text(
            instance=parsed.instance,
            remote_jid=parsed.remote_jid,
            text=result.final_reply,
        )

        storage.append_message(
            session_id=parsed.remote_jid,
            role="assistant",
            content=result.final_reply,
            message_type="text",
        )
        storage.mark_message_processed(parsed.message_id, parsed.remote_jid)
        return JSONResponse(
            {
                "status": "ok",
                "message_type": parsed.message_type,
                "handoff_requested": result.handoff_requested,
                "booking_saved": result.booking_saved,
            }
        )
    except Exception as exc:
        logger.exception("Webhook processing failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _resolve_incoming_text(parsed, trace_id: str | None) -> str:
    if parsed.message_type == "text":
        return parsed.text or ""

    if parsed.message_type == "audio":
        base64_audio, mime_type = _normalize_base64_media(
            parsed.audio_base64,
            parsed.audio_mime_type,
        )
        if not base64_audio:
            media_payload = await evolution.get_media_base64(
                parsed.instance,
                parsed.message_id,
                convert_to_mp4=True,
            )
            base64_audio, mime_type = _extract_base64_media(
                media_payload,
                "audio/mp4",
            )
        if not base64_audio:
            logger.warning("Audio message %s did not include base64 media", parsed.message_id)
            return ""
        return await media.transcribe_audio_with_trace(
            base64_audio=base64_audio,
            mime_type=mime_type,
            trace_id=trace_id,
            session_id=parsed.remote_jid,
            user_id=parsed.remote_jid,
        )

    if parsed.message_type == "image":
        base64_image, mime_type = _normalize_base64_media(
            parsed.image_base64,
            parsed.image_mime_type,
        )
        if not base64_image:
            media_payload = await evolution.get_media_base64(
                parsed.instance,
                parsed.message_id,
                convert_to_mp4=False,
            )
            base64_image, mime_type = _extract_base64_media(
                media_payload,
                mime_type or "image/jpeg",
            )
        if not base64_image:
            logger.warning("Image message %s did not include base64 media", parsed.message_id)
            return parsed.image_caption or ""
        return await media.analyze_image_with_trace(
            base64_image=base64_image,
            mime_type=mime_type,
            caption=parsed.image_caption,
            trace_id=trace_id,
            session_id=parsed.remote_jid,
            user_id=parsed.remote_jid,
        )

    return ""
