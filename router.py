from llama_index.core.llms import ChatMessage


def route_query(llm, query: str):
    prompt = f"""
You are a classifier.

Decide the type of the question based ONLY on internal project documents.

Return ONLY one word:
- structured
- semantic
- out_of_scope

Question: {query}
"""

    response = llm.chat([
        ChatMessage(role="user", content=prompt)
    ])

    return response.message.content.strip().lower()