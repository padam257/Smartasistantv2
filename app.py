import streamlit as st
import os
import re

# -------------------------------
# REMOVE CORPORATE PROXIES
# -------------------------------
for proxy in [
    "HTTP_PROXY", "HTTPS_PROXY",
    "http_proxy", "https_proxy",
    "ALL_PROXY", "all_proxy"
]:
    os.environ.pop(proxy, None)

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
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
)

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
AZURE_BLOB_CONTAINER_NAME = os.getenv(
    "AZURE_BLOB_CONTAINER_NAME", "smartassistant-index"
)

if not all([
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_ADMIN_KEY,
    AZURE_SEARCH_INDEX_NAME,
    AZURE_BLOB_CONNECTION_STRING
]):
    st.error("🚨 Missing required environment variables.")
    st.stop()

# -------------------------------
# CLIENT INIT
# -------------------------------
blob_service_client = BlobServiceClient.from_connection_string(
    AZURE_BLOB_CONNECTION_STRING
)
blob_container_client = blob_service_client.get_container_client(
    AZURE_BLOB_CONTAINER_NAME
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
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
# PROMPT
# -------------------------------
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant that answers questions strictly using the provided SOP context.

Rules:
- If the answer is not explicitly stated, you MAY estimate or infer.
- If you estimate, clearly say it is an estimation.
- If the context does not contain relevant information, reply exactly:
  "No information available in SOP documents."

Context:
{context}

Question:
{question}

Answer:
"""
)

# -------------------------------
# UI
# -------------------------------
st.title("🤖 SmartAssistantApp – SOP GenAI")

# -------------------------------
# UPLOAD SECTION
# -------------------------------
st.header("📄 Upload SOP Document")
uploaded_file = st.file_uploader(
    "Upload SOP (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"]
)

if uploaded_file:
    file_name = uploaded_file.name
    ext = file_name.split(".")[-1].lower()
    local_path = f"/tmp/{file_name}"

    with open(local_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    blob_container_client.upload_blob(
        file_name, uploaded_file, overwrite=True
    )

    if ext == "pdf":
        loader = PyPDFLoader(local_path)
    elif ext == "txt":
        loader = TextLoader(local_path)
    else:
        loader = UnstructuredFileLoader(local_path)

    docs_raw = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    docs = splitter.split_documents(docs_raw)

    for d in docs:
        d.metadata = {"file_name": file_name}

    vectorstore.add_documents(docs)
    st.success(f"Uploaded & indexed `{file_name}`")

# -------------------------------
# FILE LIST + DELETE
# -------------------------------
st.header("📂 Available SOP Files")
blobs = [b.name for b in blob_container_client.list_blobs()]

if blobs:
    delete_file = st.selectbox("Select SOP to delete", [""] + blobs)
    if st.button("🗑️ Delete SOP") and delete_file:
        # delete from blob
        blob_container_client.delete_blob(delete_file)

        # delete from search index
        results = search_client.search(
            search_text="*",
            filter=f"file_name eq '{delete_file}'",
            select=["id"]
        )
        ids = [{"id": r["id"]} for r in results]
        if ids:
            search_client.delete_documents(ids)

        st.success(f"Deleted `{delete_file}`")
        st.experimental_rerun()
else:
    st.write("No documents uploaded.")

# -------------------------------
# QUERY SECTION
# -------------------------------
st.header("🔍 Query SOP Documents")
scope = st.selectbox("Search Scope:", ["All Documents"] + blobs)
question = st.text_input("Enter your question:")

if st.button("Run Query"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching SOPs…"):
        retriever = vectorstore.as_retriever()
        retriever.k = 5   # ✅ FIX (ONLY place k is set)

        if scope != "All Documents":
            retriever.search_kwargs["filter"] = (
                f"file_name eq '{scope}'"
            )

        docs = retriever.get_relevant_documents(question)

    if not docs:
        st.subheader("📝 Answer")
        st.write("No information available in SOP documents.")
        st.stop()

    context = "\n\n".join(d.page_content for d in docs)

    answer = llm.invoke(
        RAG_PROMPT.format(
            context=context,
            question=question
        )
    ).content

    st.subheader("📝 Answer")
    st.write(answer)

    st.subheader("📌 Source Chunks")
    for d in docs:
        st.write(d.page_content[:500])
