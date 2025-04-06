import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from tempfile import NamedTemporaryFile



st.set_page_config(page_title=" RAG Chatbot", layout="wide")
st.title(" RAG-based Document Chatbot (FAISS + Ollama + Mistral)")

# Embedding model (HuggingFace)
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

# LLM from Ollama
llm = ChatOllama(model="mistral", temperature=0.7)

# ----------------- FILE HANDLING -----------------

uploaded_files = st.file_uploader("Upload one or more documents", type=["pdf", "txt", "docx"], accept_multiple_files=True)

documents = []
if uploaded_files:
    for file in uploaded_files:
        with NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(tmp_path)
        else:
            st.error("Unsupported file type.")
            continue

        docs = loader.load()
        documents.extend(docs)

# ----------------- EMBEDDING + VECTOR DB -----------------

if documents:
    st.success(f"{len(documents)} documents loaded. Embedding and indexing...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Use FAISS vector database
    vectordb = FAISS.from_documents(chunks, embedding_model)

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # ----------------- PROMPT TEMPLATE -----------------

    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
     You are a helpful assistant. Use the following context to answer the question

     Please answer the following question using clear, structured output:
     - Use bullet points for lists.
     - Use tables when appropriate.
     - Use well-formatted paragraphs for explanations.

     If the answer is not in the context don't answer, say "Sorry, I couldn't find the answer in the documents".

        Context:
        {context}

        Question:
        {question}
        """
    )

    # ----------------- QA CHAIN -----------------

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt}
    )

    # ----------------- CHAT UI -----------------

    query = st.text_input("Ask a question about your documents:")

    if query:
        with st.spinner("Answering..."):
            response = qa.invoke(query)

            st.markdown("### Answer")
            st.write(response["result"])

            with st.expander(" Source Documents"):
                for i, doc in enumerate(response["source_documents"]):
                    st.markdown(f"**Document {i+1}** — {doc.metadata.get('source', 'N/A')}")
                    st.write(doc.page_content[:500] + "...")
else:
    st.info("Upload documents to get started.")
