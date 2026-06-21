# RAG LlamaIndex Agent

## Overview

This project implements a Retrieval-Augmented Generation (RAG) agent using LlamaIndex.  
It processes user questions, determines the correct data source, retrieves relevant context, and generates an answer using an LLM.

The system is designed as an event-driven workflow with clear separation between routing, retrieval, validation, and generation.

---

## Key Capabilities

- Intelligent query routing (structured / semantic / out-of-scope)
- Retrieval from vector database (Pinecone)
- Extraction and use of structured JSON knowledge
- Confidence scoring for retrieved results
- LLM-based answer generation grounded in context only

---

## System Components

### Query Router
Classifies each user question into one of three paths:
- Structured data lookup
- Semantic document search
- Out-of-scope rejection

### Retrieval Layer
Fetches relevant information from:
- Pinecone vector store
- Structured knowledge base (JSON)

### Validation Layer
- Filters irrelevant results
- Computes confidence score
- Ensures minimum context quality before generation

### Generation Layer
- Uses LLM to produce final answer
- Answers are strictly based on retrieved context

---

## Setup Instructions

### Install dependencies
```bash
uv sync
Environment variables

Create a .env file:
COHERE_API_KEY=your_api_key
PINECONE_API_KEY=your_api_key

Running the Project
uv run app.py

After running, a local Gradio interface will be available (e.g. http://127.0.0.1:7860).

Project Structure
app.py                 # Application entry point
workflow.py           # Core RAG workflow (event-driven pipeline)
router.py             # Query classification logic
extractor.py          # Document structured data extraction
structured_store.py   # JSON structured knowledge loader
models/               # Data schemas (Pydantic models)

Example Queries
Information retrieval
How do I install dependencies?
What is Bun in this project?
How does the system work?
Structured data queries
Show me all decisions
List system rules
Out-of-scope handling
Questions outside project scope are rejected or redirected

Architecture Flow

User Input
→ Router (classification)
→ Retrieval (vector / structured)
→ Validation (filter + scoring)
→ LLM Generation
→ Final Answer