import streamlit as st
import os
import uuid

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
from langchain_community.vectorstores.azuresearch import AzureSearch
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

# -------------------------------
# VECTOR STORE (CRITICAL FIX)
# Explicit field mapping → prevents JSONDecodeError
# -------------------------------
vectorstore = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_ADMIN_KEY,
    index_name=AZURE_SEARCH_INDEX_NAME,
    embedding_function=embeddings.embed_query,
    fields={
        "id": "id",
        "content": "content",
        "content_vector": "content_vector",
        "file_name": "file_name"
    }
)

# -------------------------------
# PROMPT (RAG)
# -------------------------------
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer the question strictly using the SOP content below.
If the answer is not present, reply exactly:
"No information available in SOP documents."

SOP Content:
{context}

Question:
{question}

Answer:
"""
)

# -------------------------------
# SEARCH HELPER (NO threshold, NO out-of-scope logic)
# -------------------------------
def vector_search(query, scope):
    filters = None
    if scope != "All Documents":
        filters = f"file_name eq '{scope}'"

    docs = vectorstore.similarity_search(
        query=query,
        k=6,
        filters=filters
    )

    # 🔧 FIX: remove empty / invalid chunks
    return [d for d in docs if d.page_content and d.page_content.strip()]

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

    # 🔧 FIX: ensure id + correct metadata
    for d in docs:
        d.metadata = {
            "id": str(uuid.uuid4()),
            "file_name": file.name
        }

    vectorstore.add_documents(docs)
    st.success(f"Indexed {file.name}")

# -------------------------------
# FILE LIST + DELETE (FIXED)
# -------------------------------
st.header("📂 SOP Files")
files = [b.name for b in container.list_blobs()]
delete_file = st.selectbox("Select file to delete", [""] + files)

if st.button("🗑️ Delete Selected File") and delete_file:
    container.delete_blob(delete_file)

    # 🔧 FIX: delete by KEY FIELD (id)
    results = search_client.search(
        search_text="*",
        filter=f"file_name eq '{delete_file}'",
        select=["id"]
    )

    ids = [{"id": r["id"]} for r in results]

    if ids:
        search_client.delete_documents(documents=ids)
        st.success(f"Deleted {len(ids)} chunks from {delete_file}")
    else:
        st.warning("No indexed chunks found for this file")

    st.stop()

# -------------------------------
# QUERY + OUTPUT SECTION
# -------------------------------
st.header("🔍 Query SOPs")

scope = st.selectbox("Search Scope", ["All Documents"] + files)
question = st.text_input("Enter your question")

if st.button("Run Query") and question:
    with st.spinner("Searching SOPs…"):
        docs = vector_search(question, scope)

    st.subheader("📝 Answer")

    if not docs:
        st.write("No information available in SOP documents.")
    else:
        context = "\n\n".join(d.page_content for d in docs)

        response = llm.invoke(
            RAG_PROMPT.format(context=context, question=question)
        )

        st.write(response.content)

        st.subheader("📌 Source Chunks")
        for d in docs:
            st.write(d.page_content[:400])

# -------------------------------
# RESET
# -------------------------------
if st.button("Reset"):
    st.session_state.clear()
    st.rerun()

