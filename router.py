from llama_index.core.llms import ChatMessage


ALLOWED = {"structured", "semantic", "out_of_scope"}


async def route_query(llm, query: str) -> str:
    prompt = f"""
You are a strict classifier.

Classify the user question into EXACTLY one category:

1. structured → questions about installation, setup, commands, project usage
2. semantic → general questions answered from documents
3. out_of_scope → unrelated questions

Rules:
- Return ONLY one word
- No explanations
- No punctuation
- No extra text

Question:
{query}
"""

    response = await llm.achat([
        ChatMessage(role="user", content=prompt)
    ])

    result = response.message.content.strip().lower()

    if result not in ALLOWED:
        return "semantic"   # fallback בטוח

    return result