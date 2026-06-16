import os
import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
load_dotenv()

from netfree_unstrict_ssl import unstrict_ssl
unstrict_ssl()

from llama_index.core.workflow import (
    Workflow, step, Context, StartEvent, StopEvent, Event
)

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from llama_index.embeddings.cohere import CohereEmbedding

from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex


# 📦 EVENTS (חובה לפי הוולידציה)
class DocumentsEvent(Event):
    documents: list


class NodesEvent(Event):
    nodes: list


# 🚀 WORKFLOW
class PrepareRAGWorkflow(Workflow):

    # 🔵 Load documents
    @step
    def load_documents(self, ctx: Context, ev: StartEvent) -> DocumentsEvent:

        reader = SimpleDirectoryReader(input_dir="KIRO-STEERING")
        documents = reader.load_data()

        print(f"Loaded documents: {len(documents)}")

        return DocumentsEvent(documents=documents)

    # 🟡 Chunking
    @step
    def chunk_documents(self, ctx: Context, ev: DocumentsEvent) -> NodesEvent:

        parser = SentenceSplitter(
            chunk_size=500,
            chunk_overlap=20
        )

        nodes = parser.get_nodes_from_documents(
            documents=ev.documents,
            show_progress=True
        )

        print(f"Created chunks: {len(nodes)}")

        return NodesEvent(nodes=nodes)

    # 🔴 Build index
    @step
    def build_index(self, ctx: Context, ev: NodesEvent) -> StopEvent:

        embed_model = CohereEmbedding(
            api_key=os.getenv("COHERE_API_KEY"),
            model_name="embed-english-v3.0",
            input_type="search_document",
        )

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        pinecone_index = pc.Index("kiro-steering")

        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            namespace="kiro-steering"
        )

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        VectorStoreIndex.from_documents(
            ev.nodes,
            storage_context=storage_context,
            embed_model=embed_model
        )

        print("Index successfully built")

        return StopEvent(result={"status": "ready"})


# 🚀 RUN
if __name__ == "__main__":
    import asyncio

    async def main():
        workflow = PrepareRAGWorkflow()
        await workflow.run()

    asyncio.run(main())