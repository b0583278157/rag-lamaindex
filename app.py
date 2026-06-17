from netfree_unstrict_ssl import unstrict_ssl
unstrict_ssl()

import os
import gradio as gr
from dotenv import load_dotenv
load_dotenv()

from workflow import RAGWorkflow
from rag_core import get_pipeline

# ======================
# PIPELINE
# ======================
pipeline = get_pipeline()

# ======================
# WORKFLOW (חד-פעמי)
# ======================
workflow = RAGWorkflow(pipeline)


# ======================
# CHAT FUNCTION
# ======================
async def chat(message, history):
    try:
        result = await workflow.run(query=message)
        return str(result)

    except Exception as e:
        print(f"ERROR: {e}")
        return "שגיאה בעיבוד השאלה"


# ======================
# UI
# ======================
demo = gr.ChatInterface(
    fn=chat,
    title="📚 KIRO RAG Chat"
)

demo.launch()