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

from llama_index.core.llms import ChatMessage, MessageRole

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

    # START
    @step
    def start(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        return QueryEvent(query=ev.query, attempt=0)

    # ROUTE + RETRIEVE
    @step
    def retrieve(self, ctx: Context, ev: QueryEvent)-> Union[RetrieveEvent, GenerateEvent, StopEvent]:

        route = route_query(self.llm, ev.query)

        if route == "out_of_scope":
            return StopEvent(result="אין לי אפשרות לספק מידע זה.")

        if route == "structured":

            nodes = (
                self.structured_data.get("decisions", []) +
                self.structured_data.get("rules", []) +
                self.structured_data.get("warnings", [])
            )

            return GenerateEvent(
                query=ev.query,
                nodes=nodes,
                attempt=ev.attempt
            )

        nodes = self.retriever.retrieve(ev.query)

        return RetrieveEvent(
            query=ev.query,
            nodes=nodes,
            attempt=ev.attempt
        )

    # VALIDATE
    @step
    def validate(self, ctx: Context, ev: RetrieveEvent) -> ValidateEvent:

        nodes = [n for n in ev.nodes if n]

        if not nodes:
            return ValidateEvent(
                query=ev.query,
                nodes=[],
                is_valid=False,
                confidence=0.0,
                attempt=ev.attempt,
                reason="no results"
            )

        avg_score = sum(getattr(n, "score", 1.0) for n in nodes) / len(nodes)

        confidence = min(avg_score, 1.0)

        return ValidateEvent(
            query=ev.query,
            nodes=nodes,
            is_valid=confidence > 0.3,
            confidence=confidence,
            attempt=ev.attempt,
            reason="ok"
        )

    # DECIDE
    @step
    def decide(
        self,
        ctx: Context,
        ev: ValidateEvent
    ) -> Union[RetrieveEvent, GenerateEvent]:

        if not ev.is_valid and ev.attempt < 2:
            return RetrieveEvent(
                query=ev.query,
                nodes=self.retriever.retrieve(ev.query),
                attempt=ev.attempt + 1
            )

        return GenerateEvent(
            query=ev.query,
            nodes=ev.nodes,
            attempt=ev.attempt
        )

    # GENERATE
    @step
    def generate(self, ctx: Context, ev: GenerateEvent) -> StopEvent:

        context_text = "\n\n".join(
                        n.get_content()[:300] if hasattr(n, "get_content") else str(n)[:300]
                        for n in ev.nodes[:3]
                    )

        prompt = f"""
ענה  בעברית בלבד.

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

        answer = response.message.content

        if "not found" in answer.lower():
            return StopEvent(result="לא נמצא מידע.")

        return StopEvent(result=answer)