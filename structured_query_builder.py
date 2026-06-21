def build_query(llm, question: str):
    prompt = f"""
Convert this question into structured query JSON.

Schema types:
- rules
- decisions
- warnings

Return JSON only.

Question:
{question}
"""

    return llm.complete(prompt).text