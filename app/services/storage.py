from __future__ import annotations

import csv
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

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
