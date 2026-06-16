from netfree_unstrict_ssl import unstrict_ssl
unstrict_ssl()
import os
import gradio as gr
from dotenv import load_dotenv
load_dotenv()

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from rag_core import get_pipeline

pipeline = get_pipeline()
retriever = pipeline["retriever"]
llm = pipeline["llm"]

response_synthesizer = get_response_synthesizer(llm=llm)
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    
    response_synthesizer=response_synthesizer,
)


def chat(message, history):
    nodes = retriever.retrieve(message)
    print("NODES:", len(nodes))

    response = response_synthesizer.synthesize(
        query=message,
        nodes=nodes
    )

    return str(response)


demo = gr.ChatInterface(
    fn=chat,
    title="📚 KIRO RAG Chat"
)

demo.launch()