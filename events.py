from llama_index.core.workflow import Event

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


class ResponseEvent(Event):
    response: str

class ResponseEvent(Event):
    response: str