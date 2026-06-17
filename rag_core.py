
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


def get_pipeline():

    llm = Cohere(
    api_key=os.getenv("COHERE_API_KEY"),
    model="command-r-08-2024"
)
    print(type(llm))

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

    retriever = index.as_retriever(similarity_top_k=5)

    return {
        "llm": llm,
        "retriever": retriever
    }