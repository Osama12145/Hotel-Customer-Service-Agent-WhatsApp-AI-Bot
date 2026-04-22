from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ParsedIncomingMessage(BaseModel):
    instance: str
    message_id: str
    remote_jid: str
    push_name: str | None = None
    message_type: Literal["text", "audio", "image", "unsupported"]
    text: str | None = None
    audio_base64: str | None = None
    audio_mime_type: str | None = None
    image_base64: str | None = None
    image_mime_type: str | None = None
    image_caption: str | None = None
    raw_payload: dict


class BookingDetails(BaseModel):
    guest_name: str | None = None
    phone_number: str | None = None
    check_in_date: str | None = None
    check_out_date: str | None = None
    room_type: str | None = None
    adults: int | None = None
    children: int | None = None
    special_requests: str | None = None
    confirmed_by_guest: bool = False


class AgentDecision(BaseModel):
    intent: Literal["faq", "booking", "handoff", "other"] = "faq"
    should_handoff: bool = False
    handoff_reason: str | None = None
    reply_text: str
    booking_details: BookingDetails = Field(default_factory=BookingDetails)
    booking_ready_for_save: bool = False
    needs_more_booking_info: bool = False
    missing_booking_fields: list[str] = Field(default_factory=list)


@dataclass
class HotelAgentState:
    session_id: str
    incoming_text: str
    customer_name: str | None
    original_message_type: str
    message_id: str
    raw_payload: dict
    chat_history: list[dict] = field(default_factory=list)
    retrieved_context: str = ""
    decision: AgentDecision | None = None
    final_reply: str = ""
    handoff_requested: bool = False
    handoff_reason: str | None = None
    booking_saved: bool = False
    trace_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
