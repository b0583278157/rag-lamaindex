
from netfree_unstrict_ssl import unstrict_ssl
unstrict_ssl()
import os
from dotenv import load_dotenv
load_dotenv()

from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from extractor import enrich_source,  load_documents, extract_structured_data
from models.schema import KnowledgeData, Rule, Decision, WarningItem
from datetime import datetime
from structured_store import load_structured_data

def get_pipeline():

    llm = Cohere(
    api_key=os.getenv("COHERE_API_KEY"),
    model="command-r-08-2024"
)
    print(type(llm))
    structured_data = load_structured_data()

    embed_model = CohereEmbedding(
        api_key=os.getenv("COHERE_API_KEY"),
        model_name="embed-english-v3.0",
        input_type="search_document",
    )

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index("kiro-steering")

    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace="kiro-steering"
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )

    retriever = index.as_retriever(similarity_top_k=2)

    return {
        "llm": llm,
        "retriever": retriever
    }
import json
from extractor import load_documents, extract_structured_data




def build_knowledge_base():
    llm = get_pipeline()["llm"]
    docs = load_documents()

    decisions = []
    rules = []
    warnings = []

    for doc in docs:
        result = extract_structured_data(llm, doc.text)

        rules.extend([r.model_dump() for r in result.rules])
        decisions.extend([d.model_dump() for d in result.decisions])
        warnings.extend([w.model_dump() for w in result.warnings])

    data_dict = {
        "decisions": decisions,
        "rules": rules,
        "warnings": warnings
    }

    data_dict["validation_report"] = validate_knowledge_base(data_dict)

    with open("knowledge_data.json", "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)

    return data_dict
def validate_knowledge_base(data: dict):
    rules = data.get("rules", [])
    decisions = data.get("decisions", [])
    warnings = data.get("warnings", [])

    return {
        "rules_count": len(rules),
        "decisions_count": len(decisions),
        "warnings_count": len(warnings),
        "missing_source": sum(1 for r in rules if not r.get("source")),
        "status": "ok" if len(rules) > 0 else "warning"
    }
def build_sources(docs):
    files = []

    for doc in docs:
        files.append({
            "path": doc.metadata.get("file_path"),
            "last_modified": doc.metadata.get("last_modified_date")
        })

    return [
        {
            "tool": "kiro-steering",
            "root_path": "KIRO-STEERING",
            "files": files
        }
    ]