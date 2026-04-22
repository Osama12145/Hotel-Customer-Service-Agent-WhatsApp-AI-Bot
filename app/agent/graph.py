from __future__ import annotations

import json
import logging

from langgraph.graph import END, StateGraph

from app.agent.prompts import CLASSIFIER_SYSTEM_PROMPT, build_hotel_system_prompt
from app.models import AgentDecision, HotelAgentState
from app.services.knowledge import KnowledgeService
from app.services.langfuse_service import LangfuseService
from app.services.openrouter import OpenRouterService
from app.services.storage import StorageService


logger = logging.getLogger(__name__)


class HotelAgent:
    """
    LangGraph-based hotel assistant.

    The graph is intentionally explicit and heavily commented so the flow is easy to
    review and evolve:
    1. Load hotel knowledge relevant to the message.
    2. Ask the model to decide whether this is FAQ, booking, or handoff.
    3. Route to the final node that may save a booking or create a handoff record.
    """

    def __init__(self) -> None:
        self.storage = StorageService()
        self.knowledge = KnowledgeService()
        self.openrouter = OpenRouterService()
        self.langfuse = LangfuseService()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(HotelAgentState)
        graph.add_node("retrieve_context", self.retrieve_context_node)
        graph.add_node("decide", self.decide_node)
        graph.add_node("save_booking", self.save_booking_node)
        graph.add_node("save_handoff", self.save_handoff_node)
        graph.add_node("finalize", self.finalize_node)

        graph.set_entry_point("retrieve_context")
        graph.add_edge("retrieve_context", "decide")
        graph.add_conditional_edges(
            "decide",
            self.route_after_decision,
            {
                "save_booking": "save_booking",
                "save_handoff": "save_handoff",
                "finalize": "finalize",
            },
        )
        graph.add_edge("save_booking", "finalize")
        graph.add_edge("save_handoff", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def retrieve_context_node(self, state: HotelAgentState) -> HotelAgentState:
        state.retrieved_context = self.knowledge.retrieve(state.incoming_text)
        return state

    async def decide_node(self, state: HotelAgentState) -> HotelAgentState:
        history_text = "\n".join(
            f"{item['role']}: {item['content']}" for item in state.chat_history[-8:]
        )
        system_prompt = build_hotel_system_prompt(state.retrieved_context)
        user_prompt = f"""
سياق المحادثة السابقة:
{history_text or "لا يوجد سجل سابق."}

رسالة العميل الحالية:
{state.incoming_text}

أولًا فكّر بناءً على سياسة الفندق التالية:
{system_prompt}

ثم أعد النتيجة بصيغة JSON فقط حسب المخطط المطلوب.
""".strip()

        raw_decision = await self.openrouter.complete_text(
            system_prompt=CLASSIFIER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            trace_id=getattr(state, "trace_id", None),
            generation_name="agent-decision",
            session_id=state.session_id,
            user_id=state.session_id,
            metadata={
                "message_type": state.original_message_type,
                "message_id": state.message_id,
            },
        )
        decision = AgentDecision.model_validate(json.loads(raw_decision))
        state.decision = decision
        state.final_reply = decision.reply_text
        state.handoff_requested = decision.should_handoff
        state.handoff_reason = decision.handoff_reason
        return state

    def route_after_decision(self, state: HotelAgentState) -> str:
        if state.decision and state.decision.booking_ready_for_save:
            return "save_booking"
        if state.decision and state.decision.should_handoff:
            return "save_handoff"
        return "finalize"

    def save_booking_node(self, state: HotelAgentState) -> HotelAgentState:
        if state.decision:
            self.storage.save_booking_request(state.session_id, state.decision)
            state.booking_saved = True
        return state

    def save_handoff_node(self, state: HotelAgentState) -> HotelAgentState:
        if state.decision and state.decision.handoff_reason:
            self.storage.save_handoff_request(
                session_id=state.session_id,
                reason=state.decision.handoff_reason,
                last_user_message=state.incoming_text,
            )
        return state

    def finalize_node(self, state: HotelAgentState) -> HotelAgentState:
        # This node exists so we have one clear place for any post-processing later:
        # tone adjustments, response shortening, metadata, audit tags, etc.
        return state

    async def run(self, state: HotelAgentState) -> HotelAgentState:
        logger.info("Running agent for session %s", state.session_id)
        result = await self.graph.ainvoke(state)
        self.langfuse.update_trace(
            trace_id=getattr(result, "trace_id", None),
            output_payload={
                "final_reply": result.final_reply,
                "handoff_requested": result.handoff_requested,
                "booking_saved": result.booking_saved,
            },
            metadata={
                "handoff_reason": result.handoff_reason,
            },
        )
        self.langfuse.flush()
        return result
