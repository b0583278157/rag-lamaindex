from typing import List, Union
import json
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

        nodes = self.retriever.retrieve(ev.query)

        return RetrieveEvent(
            query=ev.query,
            nodes=nodes,
            attempt=ev.attempt
        )

    # -------------------------
    # VALIDATE
    # -------------------------
    @step
    def validate(self, ctx:Context, ev: RetrieveEvent) -> ValidateEvent:

        nodes = ev.nodes

        if not nodes:
            return ValidateEvent(
                query=ev.query,
                nodes=[],
                is_valid=False,
                confidence=0.0,
                attempt=ev.attempt
            )

        confidence = min(len(nodes) / 5, 1.0)

        return ValidateEvent(
            query=ev.query,
            nodes=nodes,
            is_valid=confidence > 0.4,
            confidence=confidence,
            attempt=ev.attempt
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
    def generate(self, ctx:Context, ev: GenerateEvent) -> StopEvent:

        context_text = "\n\n".join(str(n) for n in ev.nodes)

        prompt = f"""
You are a strict QA system.

Use ONLY the provided context.

Return JSON:
{{
  "answer": "...",
  "confidence": 0-1,
  "sources": []
}}

If not found → answer = "Not found"

Context:
{context_text}

Question:
{ev.query}
"""

        response = self.llm.chat([
            ChatMessage(
                role=MessageRole.USER,
                content=prompt
            )
        ])

        raw = response.message.content

        try:
            data = json.loads(raw)
            result = AnswerSchema(**data)
        except Exception:
            result = AnswerSchema(
                answer=raw,
                confidence=0.0,
                sources=[]
            )

        # אם לא נמצא → תשובה נקייה
        if result.answer == "Not found":
            return StopEvent(result="איני יכול לספק מידע על שאלה זאת.")

        return StopEvent(result=result.answer)