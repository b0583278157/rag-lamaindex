from llama_index.core.base.llms.types import ChatMessage, MessageRole

def route_query(llm, query: str) -> str:
    response = llm.chat([
        ChatMessage(
            role=MessageRole.USER,
            content=f"""
Classify the query:
Return only: structured or semantic

Query: {query}
"""
        )
    ])

    result = response.message.content.strip().lower()

    if result not in ["structured", "semantic"]:
        return "semantic"

    return result