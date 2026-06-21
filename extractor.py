from llama_index.core import SimpleDirectoryReader
from models.schema import KnowledgeData
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.program import LLMTextCompletionProgram
from models.schema import Source


def load_documents():
    documents = SimpleDirectoryReader(
        input_dir="KIRO-STEERING",
        required_exts=[".md"]
    ).load_data()

    return documents


def extract_structured_data(llm, text: str):

    program = LLMTextCompletionProgram.from_defaults(
        llm=llm,
        output_cls=KnowledgeData,
        prompt_template_str="""
You are a strict structured data extractor.

Extract ONLY explicit information.

Identify:
1. Technical decisions
2. Rules and guidelines
3. Warnings, sensitive areas or things marked as "do not change"

Return data according to the schema.

Do not invent information.

Text:
{text}
""",
        verbose=True,
    )

    result = program(text=text)

    return result

def enrich_source(doc):
    return Source(
        file=doc.metadata.get("file_name"),
        line_start=None,
        line_end=None,
        anchor=None
    )