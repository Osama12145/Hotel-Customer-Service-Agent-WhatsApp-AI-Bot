from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any
from uuid import uuid4

from app.config import get_settings


logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse
except Exception:  # pragma: no cover - import safety only
    Langfuse = None


class LangfuseService:
    """Optional tracing helper.

    The service stays silent unless all Langfuse settings are provided.
    This keeps local development friction low while making production
    tracing easy to switch on.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.enabled = bool(
            settings.langfuse_host
            and settings.langfuse_public_key
            and settings.langfuse_secret_key
            and Langfuse is not None
        )
        self.client = None
        if self.enabled:
            self.client = Langfuse(
                host=settings.langfuse_host,
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
            )

    def create_trace(
        self,
        *,
        name: str,
        user_id: str,
        session_id: str,
        input_payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str | None, str | None]:
        if not self.enabled or not self.client:
            return None, None

        trace_id = str(uuid4())
        trace = self.client.trace(
            id=trace_id,
            name=name,
            user_id=user_id,
            session_id=session_id,
            input=input_payload,
            metadata=metadata or {},
        )
        return trace_id, getattr(trace, "id", trace_id)

    def log_generation(
        self,
        *,
        trace_id: str | None,
        name: str,
        model: str,
        input_payload: Any,
        output_payload: Any = None,
        metadata: dict[str, Any] | None = None,
        level: str = "DEFAULT",
        status_message: str | None = None,
    ) -> None:
        if not self.enabled or not self.client or not trace_id:
            return
        self.client.generation(
            trace_id=trace_id,
            name=name,
            model=model,
            input=input_payload,
            output=output_payload,
            metadata=metadata or {},
            level=level,
            status_message=status_message,
        )

    def update_trace(
        self,
        *,
        trace_id: str | None,
        output_payload: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or not self.client or not trace_id:
            return
        self.client.trace(
            id=trace_id,
            output=output_payload,
            metadata=metadata or {},
        )

    def flush(self) -> None:
        if not self.enabled or not self.client:
            return
        with suppress(Exception):
            self.client.flush()

