import nt
from typing import List, Union
import json
from networkx import nodes
from pydantic import BaseModel

from llama_index.core.workflow import (
    Workflow,
    step,
    StartEvent,
    StopEvent,
    Event,
    Context
)

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from router import route_query
from structured_store import load_structured_data


# =========================
# Schema
# =========================

class AnswerSchema(BaseModel):
    answer: str
    confidence: float = 0.0
    sources: List[str] = []


# =========================
# Events
# =========================

class QueryEvent(Event):
    query: str
    attempt: int = 0


class RetrieveEvent(Event):
    query: str
    nodes: list
    attempt: int


class ValidateEvent(Event):
    query: str
    nodes: list
    is_valid: bool
    confidence: float
    attempt: int
    reason: str 


class DecideEvent(Event):
    query: str
    nodes: list
    confidence: float
    attempt: int


class GenerateEvent(Event):
    query: str
    nodes: list
    attempt: int


# =========================
# Workflow
# =========================

class RAGWorkflow(Workflow):

    def __init__(self, pipeline):
        super().__init__()
        self.retriever = pipeline["retriever"]
        self.llm = pipeline["llm"]
        self.structured_data = load_structured_data()

    # -------------------------
    # START
    # -------------------------
    @step
    def start(self, ctx:Context, ev: StartEvent) -> QueryEvent:
        return QueryEvent(query=ev.query, attempt=0)

    # -------------------------
    # RETRIEVE
    # -------------Context------------
    @step
    def retrieve(self, ctx:Context, ev: QueryEvent) -> RetrieveEvent:
        
        route = route_query(self.llm, ev.query)
        

        if route == "structured":
            return GenerateEvent(
                query=ev.query,
                nodes=self.structured_data,
                attempt=ev.attempt
            )
        print("RETRIEVE QUERY:", ev.query)
        nodes = self.retriever.retrieve(ev.query)
        print("NODES FOUND:", len(nodes))
        return RetrieveEvent(
            query=ev.query,
            nodes=nodes,
            attempt=ev.attempt
        )

    # -------------------------
    # VALIDATE
    # -------------------------
    @step
    def validate(self, ctx: Context, ev: RetrieveEvent) -> ValidateEvent:

        nodes = [
            n for n in ev.nodes
            if n is not None and str(n).strip()
        ]

        # קלט ריק
        if not ev.query or not ev.query.strip():
            return ValidateEvent(
                query=ev.query,
                nodes=[],
                is_valid=False,
                confidence=0.0,
                attempt=ev.attempt,
                reason="empty query"
            )

        # אין תוצאות
        if not nodes:
            return ValidateEvent(
                query=ev.query,
                nodes=[],
                is_valid=False,
                confidence=0.0,
                attempt=ev.attempt,
                reason="no results"
            )

        # מעט מדי תוצאות
        if len(nodes) < 2:
            return ValidateEvent(
                query=ev.query,
                nodes=nodes,
                is_valid=False,
                confidence=0.2,
                attempt=ev.attempt,
                reason="too few results"
            )

    # חישוב confidence
        avg_score = sum(
            getattr(n, "score", 1.0)
            for n in nodes
        ) / len(nodes)

        confidence = (
            0.5 * min(len(nodes) / 5, 1.0)
            + 0.5 * avg_score
        )

        # confidence נמוך
        if confidence < 0.4:
            return ValidateEvent(
                query=ev.query,
                nodes=nodes,
                is_valid=False,
                confidence=confidence,
                attempt=ev.attempt,
                reason="low confidence"
            )

        return ValidateEvent(
            query=ev.query,
            nodes=nodes,
            is_valid=True,
            confidence=confidence,
            attempt=ev.attempt,
            reason="ok"
        )

    # -------------------------
    # DECISION ROUTING
    # -------------------------
    @step
    def decide(self, ctx:Context, ev: ValidateEvent) -> Union[RetrieveEvent, GenerateEvent]:

        # אם אין מספיק מידע → retry (עד 2 פעמים)
        if not ev.is_valid and ev.attempt < 2:

            new_nodes = self.retriever.retrieve(ev.query)

            return RetrieveEvent(
                query=ev.query,
                nodes=new_nodes,
                attempt=ev.attempt + 1
            )

        return GenerateEvent(
            query=ev.query,
            nodes=ev.nodes,
            attempt=ev.attempt
        )

    # -------------------------
    # GENERATE (LLM)
    # -------------------------
    @step
    def generate(self, ctx: Context, ev: GenerateEvent) -> StopEvent:

        context_text = "\n\n".join(
            n.get_content() if hasattr(n, "get_content") else str(n)
            for n in ev.nodes
        )

        prompt = f"""
You are a strict QA system.

Answer ONLY in Hebrew.
Do NOT use English unless the term itself is a name.

Rules:
- Plain text only
- Clear and concise
- Based only on context

Context:
{context_text}

Question:
{ev.query}

Answer:
"""

        response = self.llm.chat([
            ChatMessage(
                role=MessageRole.USER,
                content=prompt
            )
        ])

        raw = response.message.content

        try:
            result = AnswerSchema(
                    answer=raw,
                    confidence=0.0,
                    sources=[]
)
            result = AnswerSchema(**data)
        except Exception:
            result = AnswerSchema(
                answer=raw,
                confidence=0.0,
                sources=[]
            )

        if result.answer == "Not found":
            return StopEvent(result="איני יכול לספק מידע על שאלה זאת.")

        return StopEvent(result=result.answer)