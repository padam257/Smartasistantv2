import streamlit as st
import os
import re

# -------------------------------
# REMOVE PROXIES
# -------------------------------
for p in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(p, None)

# -------------------------------
# IMPORTS
# -------------------------------
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredFileLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# -------------------------------
# ENV VARIABLES
# -------------------------------
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
AZURE_BLOB_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME", "smartassistant-index")

# -------------------------------
# CLIENTS
# -------------------------------
blob_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
container = blob_client.get_container_client(AZURE_BLOB_CONTAINER_NAME)

search_client = SearchClient(
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_INDEX_NAME,
    AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
)

llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    api_version="2024-02-15-preview",
    temperature=0,
    max_tokens=600
)

embeddings = AzureOpenAIEmbeddings(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    api_version="2024-02-15-preview"
)

vectorstore = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_ADMIN_KEY,
    index_name=AZURE_SEARCH_INDEX_NAME,
    embedding_function=embeddings.embed_query
)

# -------------------------------
# PROMPT (RAG)
# -------------------------------
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant that answers strictly from SOP context.

If the context does not contain the answer, reply exactly:
"No information available in SOP documents."

Context:
{context}

Question:
{question}

Answer:
"""
)

# -------------------------------
# HELPERS
# -------------------------------
SIMILARITY_THRESHOLD = 0.55

def dedupe_docs(docs):
    seen = set()
    unique = []
    for d in docs:
        key = d.page_content[:150]
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique

def safe_vector_search(query, scope):
    try:
        if scope == "All Documents":
            results = vectorstore.similarity_search_with_score(query, k=6)
        else:
            results = vectorstore.similarity_search_with_score(
                query,
                k=6,
                filters=f"file_name eq '{scope}'"
            )
    except Exception:
        return []

    # 🔧 FIX: apply threshold ONLY for All Documents
    if scope == "All Documents":
        docs = [doc for doc, score in results if score <= SIMILARITY_THRESHOLD]
    else:
        docs = [doc for doc, _ in results]  # trust user-selected scope

    return dedupe_docs(docs)

# -------------------------------
# UI
# -------------------------------
st.title("🤖 SmartAssistant – SOP RAG")

# -------------------------------
# UPLOAD
# -------------------------------
st.header("📄 Upload SOP")
file = st.file_uploader("Upload PDF / TXT / DOCX", type=["pdf", "txt", "docx"])

if file:
    path = f"/tmp/{file.name}"
    with open(path, "wb") as f:
        f.write(file.getbuffer())

    container.upload_blob(file.name, file, overwrite=True)

    loader = (
        PyPDFLoader(path) if file.name.endswith("pdf")
        else TextLoader(path) if file.name.endswith("txt")
        else UnstructuredFileLoader(path)
    )

    docs = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=200
    ).split_documents(loader.load())

    for d in docs:
        d.metadata = {"file_name": file.name}

    vectorstore.add_documents(docs)
    st.success(f"Indexed {file.name}")

# -------------------------------
# FILE LIST + DELETE
# -------------------------------
st.header("📂 SOP Files")
files = [b.name for b in container.list_blobs()]
selected_delete = st.selectbox("Select file to delete", [""] + files)

if st.button("🗑️ Delete Selected File") and selected_delete:
    container.delete_blob(selected_delete)
    search_client.delete_documents(
        documents=[{"file_name": selected_delete}]
    )
    st.success(f"Deleted {selected_delete}")
    st.stop()

# -------------------------------
# QUERY
# -------------------------------
st.header("🔍 Query SOPs")

scope = st.selectbox("Search Scope", ["All Documents"] + files)
question = st.text_input("Enter your question")

col1, col2 = st.columns(2)

if col1.button("Run Query"):
    with st.spinner("Searching SOPs…"):
        docs = safe_vector_search(question, scope)

    if not docs:
        st.subheader("📝 Answer")
        st.write("No information available in SOP documents.")
        st.stop()

    context = "\n\n".join(d.page_content for d in docs)

    answer = llm.invoke(
        RAG_PROMPT.format(context=context, question=question)
    ).content

    st.subheader("📝 Answer")
    st.write(answer)

    st.subheader("📌 Source Chunks")
    for d in docs:
        st.write(d.page_content[:400])

if col2.button("Reset"):
    st.session_state.clear()
    st.experimental_rerun()


