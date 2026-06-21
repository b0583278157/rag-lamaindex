from llama_index.core.workflow import Event

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

class ResponseEvent(Event):
    response: str