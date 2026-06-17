from typing import Union

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from gradio.mcp import prompt
from llama_index.core.workflow import (
    Workflow,
    step,
    Context,
    StartEvent,
    StopEvent,
    Event,
)
import json
from pydantic import BaseModel
from typing import List

class AnswerSchema(BaseModel):
    answer: str
    confidence: float = 0.0
    sources: List[str] = []

# =========================
# 📦 EVENTS
# =========================

class QueryEvent(Event):
    query: str


class RetrievedEvent(Event):
    query: str
    nodes: list


class ValidateEvent(Event):
    query: str
    nodes: list
    is_valid: bool
    reason: str = ""


class GenerateEvent(Event):
    query: str
    nodes: list


# =========================
# 🚀 WORKFLOW
# =========================

class RAGWorkflow(Workflow):

    def __init__(self, pipeline):
        super().__init__()
        self.retriever = pipeline["retriever"]
        self.llm = pipeline["llm"]

    # 1️⃣ START
    @step
    def start(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        return QueryEvent(query=ev.query)

    # 2️⃣ RETRIEVE
    @step
    def retrieve(self, ctx: Context, ev: QueryEvent) -> RetrievedEvent:
        nodes = self.retriever.retrieve(ev.query)

        return RetrievedEvent(
            query=ev.query,
            nodes=nodes
        )

    # 3️⃣ VALIDATE (Event-driven validation)
    @step
    def validate(self, ctx: Context, ev: RetrievedEvent) -> ValidateEvent:

        nodes = ev.nodes

        if not nodes:
            return ValidateEvent(
                query=ev.query,
                nodes=[],
                is_valid=False,
                reason="No retrieval results"
            )

        if len(nodes) < 2:
            return ValidateEvent(
                query=ev.query,
                nodes=nodes,
                is_valid=False,
                reason="Too few results"
            )

        return ValidateEvent(
            query=ev.query,
            nodes=nodes,
            is_valid=True
        )

    # 4️⃣ ROUTING (Event-driven branching)
    @step
    def route(
        self,
        ctx: Context,
        ev: ValidateEvent
    ) -> Union[RetrievedEvent, GenerateEvent]:

        # ❌ לא תקין → retry retrieval
        if not ev.is_valid:
            nodes = self.retriever.retrieve(ev.query)

            return RetrievedEvent(
                query=ev.query,
                nodes=nodes
            )

        # ✅ תקין → המשך ליצירה
        return GenerateEvent(
            query=ev.query,
            nodes=ev.nodes
        )

    # 5️⃣ GENERATE (RAG אמיתי)
    @step
    def generate(self, ctx: Context, ev: GenerateEvent) -> StopEvent:
       
        context_text = "\n\n".join(
            [str(n) for n in ev.nodes]
        )
        
        prompt = f"""
    You are a strict QA system.

    Use ONLY the provided context.

    Return ONLY valid JSON in this format:

    {{
    "answer": "...",
    "confidence": 0-1,
    "sources": ["..."]
    }}

    Rules:
    - Answer using only the provided context.
    - If the context does not contain enough information, return "Not found".
    - Return a valid JSON object.

    Context:
    {context_text}

    Question:
    {ev.query}
    """

        print("BEFORE LLM CALL")

        response = self.llm.chat(
            [
                ChatMessage(
                    role=MessageRole.USER,
                    content=prompt
                )
            ]
        )

        raw = response.message.content
        print("RAW RESPONSE:")
        print(raw)  

        try:
            data = json.loads(raw)
            validated = AnswerSchema(**data)
            print("VALIDATED:")
            print(validated)

        except Exception:
            validated = AnswerSchema(
                answer=raw,
                confidence=0.0,
                sources=[]
            )

        print("AFTER LLM CALL")

        formatted_text = self.format_answer(validated)
        return StopEvent(result=formatted_text)

    def format_answer(self,answer: AnswerSchema) -> str:
        if answer.answer == "Not found":
            return "איני יכול לספק מידע על שאלה זאת."

        confidence_percent = int(answer.confidence * 100)

        return answer.answer
        