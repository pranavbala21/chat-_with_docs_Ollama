This project is a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload documents and ask questions about them. It uses LangChain for document processing, FAISS for vector storage, and a local Mistral LLM running via Ollama.

---

## 🚀 Features

- 📁 Upload and process PDF, TXT, and DOCX files
- 🔍 Semantic document retrieval using FAISS
- 🧠 Query answering using Mistral LLM via Ollama
- 🧾 Structured output with bullet points, tables, and paragraphs
- ✅ Local inference (no API limits or latency)

---

## 🧠 Tech Stack

| Component        | Tool/Library                |
|------------------|-----------------------------|
| Frontend         | Streamlit                   |
| Embeddings       | `intfloat/e5-small-v2`      |
| Vector Database  | FAISS                       |
| Document Parsing | LangChain Loaders           |
| LLM              | Mistral via Ollama          |

---
