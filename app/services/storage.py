from __future__ import annotations

import csv
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from app.config import get_settings
from app.models import AgentDecision


class StorageService:
    """SQLite storage for messages, dedupe, handoffs, and booking requests."""

    def __init__(self) -> None:
        settings = get_settings()
        self.db_path = Path(settings.database_url.replace("sqlite:///", ""))
        self.export_path = Path(settings.bookings_export_file)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.export_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS processed_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS booking_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    guest_name TEXT,
                    phone_number TEXT,
                    check_in_date TEXT,
                    check_out_date TEXT,
                    room_type TEXT,
                    adults INTEGER,
                    children INTEGER,
                    special_requests TEXT,
                    confirmed_by_guest INTEGER DEFAULT 0,
                    llm_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS handoff_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    last_user_message TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS message_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_id TEXT,
                    customer_name TEXT,
                    user_message TEXT NOT NULL,
                    assistant_reply TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    language TEXT,
                    intent TEXT,
                    answer_confidence TEXT,
                    response_ms INTEGER,
                    handoff_requested INTEGER DEFAULT 0,
                    booking_saved INTEGER DEFAULT 0,
                    knowledge_gap INTEGER DEFAULT 0,
                    missing_topic TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS knowledge_gaps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    normalized_question TEXT NOT NULL UNIQUE,
                    sample_question TEXT NOT NULL,
                    missing_topic TEXT,
                    suggested_knowledge_section TEXT,
                    session_id TEXT,
                    language TEXT,
                    intent TEXT,
                    occurrences INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'open',
                    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def has_processed_message(self, message_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM processed_messages WHERE message_id = ?",
                (message_id,),
            ).fetchone()
        return row is not None

    def mark_message_processed(self, message_id: str, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO processed_messages (message_id, session_id) VALUES (?, ?)",
                (message_id, session_id),
            )

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        message_type: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (session_id, role, content, message_type)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, message_type),
            )

    def get_recent_history(self, session_id: str, limit: int = 12) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, message_type, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        return [dict(row) for row in reversed(rows)]

    def save_booking_request(self, session_id: str, decision: AgentDecision) -> None:
        booking = decision.booking_details
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO booking_requests (
                    session_id, status, guest_name, phone_number, check_in_date,
                    check_out_date, room_type, adults, children, special_requests,
                    confirmed_by_guest, llm_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    "pending_staff_review",
                    booking.guest_name,
                    booking.phone_number,
                    booking.check_in_date,
                    booking.check_out_date,
                    booking.room_type,
                    booking.adults,
                    booking.children,
                    booking.special_requests,
                    int(booking.confirmed_by_guest),
                    json.dumps(decision.model_dump(mode="json"), ensure_ascii=False),
                ),
            )
        self._append_booking_to_csv(session_id, decision)

    def _append_booking_to_csv(self, session_id: str, decision: AgentDecision) -> None:
        booking = decision.booking_details
        file_exists = self.export_path.exists()
        with self.export_path.open("a", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "session_id",
                    "guest_name",
                    "phone_number",
                    "check_in_date",
                    "check_out_date",
                    "room_type",
                    "adults",
                    "children",
                    "special_requests",
                    "confirmed_by_guest",
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "session_id": session_id,
                    "guest_name": booking.guest_name,
                    "phone_number": booking.phone_number,
                    "check_in_date": booking.check_in_date,
                    "check_out_date": booking.check_out_date,
                    "room_type": booking.room_type,
                    "adults": booking.adults,
                    "children": booking.children,
                    "special_requests": booking.special_requests,
                    "confirmed_by_guest": booking.confirmed_by_guest,
                }
            )

    def save_handoff_request(self, session_id: str, reason: str, last_user_message: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO handoff_requests (session_id, reason, last_user_message)
                VALUES (?, ?, ?)
                """,
                (session_id, reason, last_user_message),
            )

    def save_message_event(
        self,
        *,
        session_id: str,
        message_id: str | None,
        customer_name: str | None,
        user_message: str,
        assistant_reply: str,
        message_type: str,
        language: str | None,
        intent: str | None,
        answer_confidence: str | None,
        response_ms: int | None,
        handoff_requested: bool,
        booking_saved: bool,
        knowledge_gap: bool,
        missing_topic: str | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO message_events (
                    session_id, message_id, customer_name, user_message, assistant_reply,
                    message_type, language, intent, answer_confidence, response_ms,
                    handoff_requested, booking_saved, knowledge_gap, missing_topic
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    message_id,
                    customer_name,
                    user_message,
                    assistant_reply,
                    message_type,
                    language,
                    intent,
                    answer_confidence,
                    response_ms,
                    int(handoff_requested),
                    int(booking_saved),
                    int(knowledge_gap),
                    missing_topic,
                ),
            )

    def save_knowledge_gap(
        self,
        *,
        question: str,
        missing_topic: str | None,
        suggested_knowledge_section: str | None,
        session_id: str,
        language: str | None,
        intent: str | None,
    ) -> None:
        normalized_question = self._normalize_question(question)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO knowledge_gaps (
                    normalized_question, sample_question, missing_topic,
                    suggested_knowledge_section, session_id, language, intent
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(normalized_question) DO UPDATE SET
                    occurrences = occurrences + 1,
                    missing_topic = COALESCE(excluded.missing_topic, knowledge_gaps.missing_topic),
                    suggested_knowledge_section = COALESCE(
                        excluded.suggested_knowledge_section,
                        knowledge_gaps.suggested_knowledge_section
                    ),
                    session_id = excluded.session_id,
                    language = excluded.language,
                    intent = excluded.intent,
                    last_seen_at = CURRENT_TIMESTAMP
                """,
                (
                    normalized_question,
                    question,
                    missing_topic,
                    suggested_knowledge_section,
                    session_id,
                    language,
                    intent,
                ),
            )

    def get_analytics_summary(self, days: int = 30) -> dict[str, Any]:
        since = self._since(days)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(DISTINCT session_id) AS conversations,
                    COUNT(*) AS messages,
                    COALESCE(SUM(booking_saved), 0) AS booking_leads,
                    COALESCE(SUM(handoff_requested), 0) AS handoffs,
                    COALESCE(SUM(knowledge_gap), 0) AS knowledge_gaps,
                    ROUND(AVG(response_ms), 0) AS avg_response_ms
                FROM message_events
                WHERE created_at >= ?
                """,
                (since,),
            ).fetchone()

            intents = conn.execute(
                """
                SELECT intent, COUNT(*) AS count
                FROM message_events
                WHERE created_at >= ? AND intent IS NOT NULL
                GROUP BY intent
                ORDER BY count DESC
                """,
                (since,),
            ).fetchall()

            languages = conn.execute(
                """
                SELECT language, COUNT(*) AS count
                FROM message_events
                WHERE created_at >= ? AND language IS NOT NULL
                GROUP BY language
                ORDER BY count DESC
                """,
                (since,),
            ).fetchall()

        return {
            "days": days,
            "conversations": row["conversations"] or 0,
            "messages": row["messages"] or 0,
            "booking_leads": row["booking_leads"] or 0,
            "handoffs": row["handoffs"] or 0,
            "knowledge_gaps": row["knowledge_gaps"] or 0,
            "avg_response_ms": row["avg_response_ms"] or 0,
            "intents": [dict(item) for item in intents],
            "languages": [dict(item) for item in languages],
        }

    def get_top_questions(self, days: int = 30, limit: int = 20) -> list[dict[str, Any]]:
        since = self._since(days)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT user_message, intent, language, COUNT(*) AS count
                FROM message_events
                WHERE created_at >= ?
                GROUP BY LOWER(TRIM(user_message)), intent, language
                ORDER BY count DESC, MAX(created_at) DESC
                LIMIT ?
                """,
                (since, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_recent_message_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM message_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_knowledge_gaps(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM knowledge_gaps
                ORDER BY status = 'open' DESC, occurrences DESC, last_seen_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _since(days: int) -> str:
        return (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _normalize_question(question: str) -> str:
        return " ".join(question.strip().lower().split())[:300]
