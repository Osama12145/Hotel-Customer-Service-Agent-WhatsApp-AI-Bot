from __future__ import annotations

from typing import Any

from app.models import ParsedIncomingMessage


class PayloadParser:
    """Normalizes EvolutionAPI webhook payloads into one consistent shape."""

    @staticmethod
    def parse(payload: dict[str, Any]) -> ParsedIncomingMessage:
        body = payload.get("body", payload)
        data = body["data"]
        message = data["message"]
        key = data["key"]
        message_type = data.get("messageType", "")

        parsed = ParsedIncomingMessage(
            instance=body["instance"],
            message_id=key["id"],
            remote_jid=key["remoteJid"],
            from_me=bool(key.get("fromMe", False)),
            push_name=data.get("pushName"),
            message_type="unsupported",
            raw_payload=body,
        )

        if message_type == "conversation":
            parsed.message_type = "text"
            parsed.text = message.get("conversation", "")
            return parsed

        if message_type == "audioMessage":
            audio = message.get("audioMessage", {})
            parsed.message_type = "audio"
            parsed.audio_base64 = message.get("base64")
            parsed.audio_mime_type = audio.get("mimetype")
            return parsed

        if message_type == "imageMessage":
            image = message.get("imageMessage", {})
            parsed.message_type = "image"
            parsed.image_base64 = message.get("base64")
            parsed.image_mime_type = image.get("mimetype")
            parsed.image_caption = image.get("caption")
            return parsed

        return parsed
