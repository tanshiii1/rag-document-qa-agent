# 🔬 Research Paper Q&A Agent

A retrieval-augmented generation (RAG) pipeline that lets you upload any research paper (PDF) and ask questions about it, with answers grounded strictly in the document's actual content — not the model's general knowledge.

## What it does

- Upload a PDF research paper
- Ask questions in plain English (e.g. "What dataset did this paper use?")
- Get answers pulled directly from the document, with a built-in refusal rule: if the answer isn't in the paper, the model says so instead of guessing or hallucinating

## How it works

1. **Chunking** — the PDF is split into overlapping text segments using LangChain's recursive text splitter
2. **Embedding** — each chunk is converted into a vector using a local HuggingFace sentence-transformer model (`all-MiniLM-L6-v2`), so no external API is needed for this step
3. **Indexing** — chunks are stored in a local FAISS vector store for fast similarity search
4. **Retrieval** — when a question is asked, the most relevant chunks are retrieved from the index
5. **Generation** — the retrieved context + question are passed to an LLM (Llama 3.3 70B via Groq) with a strict system prompt instructing it to answer only from the given context

## Tech stack

- **Python**
- **LangChain** — orchestration of the RAG pipeline
- **Groq API (Llama 3.3 70B)** — LLM for answer generation
- **HuggingFace Sentence Transformers** — local embedding generation
- **FAISS** — vector similarity search
- **Streamlit** — interactive web interface

## Setup

1. Install dependencies:
```pip install streamlit langchain langchain-community langchain-groq langchain-huggingface faiss-cpu python-dotenv pypdf```
2. Add your Groq API key to a `.env` file (get a free key at https://console.groq.com/keys):
```GROQ_API_KEY=your_key_here```
3. Run the app:
```streamlit run app.py```

## Notes

Includes error handling for upstream LLM provider failures (timeouts, rate limits, service overload), so the app fails gracefully with a clear message rather than hanging indefinitely.
