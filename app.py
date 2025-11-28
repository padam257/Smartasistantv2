# app.py
import os
import openai
import streamlit as st
from pathlib import Path
import tempfile

from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient

# lightweight langchain pieces (only for embeddings + vectorstore wrapper)
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch

# basic text splitter & loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader

# -------------------------
# Configuration (env vars)
# -------------------------
openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")                # chat / completion deployment
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")  # embedding deployment
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
AZURE_BLOB_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME", "smartassistant-sops")

# -------------------------
# Basic validation
# -------------------------
required = {
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_DEPLOYMENT_NAME": AZURE_OPENAI_DEPLOYMENT_NAME,
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    "AZURE_SEARCH_ENDPOINT": AZURE_SEARCH_ENDPOINT,
    "AZURE_SEARCH_ADMIN_KEY": AZURE_SEARCH_ADMIN_KEY,
    "AZURE_SEARCH_INDEX_NAME": AZURE_SEARCH_INDEX_NAME,
    "AZURE_BLOB_CONNECTION_STRING": AZURE_BLOB_CONNECTION_STRING,
}

missing = [k for k, v in required.items() if not v]
if missing:
    st.error(f"🚨 Missing environment variables: {', '.join(missing)}")
    st.stop()

# -------------------------
# Azure clients
# -------------------------
try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
    blob_container_client = blob_service_client.get_container_client(AZURE_BLOB_CONTAINER_NAME)
except Exception as e:
    st.error(f"Failed to create Blob client: {e}")
    st.stop()

try:
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY),
    )
except Exception as e:
    st.error(f"Failed to create Search client: {e}")
    st.stop()

# -------------------------
# Embeddings & vectorstore (LangChain wrappers)
# -------------------------
# NOTE: these objects are used for embedding / retrieving docs only.
embeddings = AzureOpenAIEmbeddings(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
)

vectorstore = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_ADMIN_KEY,
    index_name=AZURE_SEARCH_INDEX_NAME,
    embedding_function=embeddings.embed_query,
)

# Helper to retrieve top-k docs for a query using the vectorstore's retriever
def retrieve_top_docs(query: str, k: int = 5):
    retriever = vectorstore.as_retriever()
    # retriever usually exposes get_relevant_documents
    try:
        docs = retriever.get_relevant_documents(query)  # default k used by retriever
    except TypeError:
        # older/newer mismatch; try get_relevant_documents with k
        docs = retriever.get_relevant_documents(query, k=k)
    # ensure we return at most k
    return docs[:k]

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="SmartAssistantApp", layout="wide")
st.title("🤖 SmartAssistantApp: SOP GenAI")
st.markdown("Upload SOPs (pdf/txt/docx), index them, and ask questions. Answers are produced using Azure OpenAI and Azure Cognitive Search vectors.")

# Upload section
st.header("📄 Upload New SOP")
uploaded_file = st.file_uploader("Upload SOP", type=["pdf", "txt", "docx"])

if uploaded_file:
    file_name = uploaded_file.name
    file_ext = file_name.split(".")[-1].lower()
    tmp_file_path = f"/tmp/{file_name}"
    with open(tmp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # upload to blob
    try:
        blob_container_client.upload_blob(file_name, uploaded_file, overwrite=True)
        st.success(f"Uploaded `{file_name}` to blob storage.")
    except Exception as e:
        st.error(f"Failed to upload to blob: {e}")
        st.stop()

    # choose loader
    if file_ext == "pdf":
        loader = PyPDFLoader(tmp_file_path)
    elif file_ext == "txt":
        loader = TextLoader(tmp_file_path)
    elif file_ext == "docx":
        loader = UnstructuredFileLoader(tmp_file_path)
    else:
        st.error("Unsupported file type")
        st.stop()

    try:
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        # convert and sanitize metadata for Azure Search
        to_index = []
        for c in chunks:
            meta = c.metadata or {}
            # minimal allowed metadata keys for index
            metadata_storage_name = meta.get("metadata_storage_name", file_name)
            page = meta.get("page", 0)
            source = meta.get("source", metadata_storage_name)

            # create simple dict object expected by vectorstore.add_documents
            # langchain's Document is not strictly necessary; AzureSearch wrapper accepts dicts too.
            # We'll pass as LangChain Document style if supported.
            doc_obj = {
                "page_content": c.page_content,
                "metadata": {
                    "metadata_storage_name": metadata_storage_name,
                    "page": int(page) if isinstance(page, (int, str)) and str(page).isdigit() else 0,
                    "source": source
                }
            }
            to_index.append(doc_obj)

        # add to vectorstore (this will create embeddings and push to Azure Search index)
        vectorstore.add_documents(to_index)
        st.success(f"Indexed `{file_name}` with {len(to_index)} chunks.")
    except Exception as e:
        st.error(f"Indexing failed: {e}")

# List blobs
st.header("📁 Available SOPs in Blob Storage")
try:
    blobs = list(blob_container_client.list_blobs())
    doc_names = [b.name for b in blobs]
except Exception as e:
    st.error(f"Failed to list blobs: {e}")
    doc_names = []

if doc_names:
    for name in doc_names:
        st.markdown(f"- {name}")
else:
    st.info("No SOPs uploaded yet.")

# Query UI
st.header("🔍 Query SOPs")
query_scope = st.selectbox("Run query on:", ["All Documents"] + doc_names, index=0)
user_query = st.text_input("Your question:")

if user_query:
    # retrieve docs (optionally filtered by metadata_storage_name)
    if query_scope != "All Documents":
        # apply simple filter by metadata name via retriever if supported by vectorstore
        # The azuresearch retriever supports search_kwargs — use as_retriever with filter if available
        try:
            retr = vectorstore.as_retriever(search_kwargs={"filter": f"metadata_storage_name eq '{query_scope}'"})
            docs = retr.get_relevant_documents(user_query)
        except Exception:
            docs = retrieve_top_docs(user_query, k=5)
    else:
        docs = retrieve_top_docs(user_query, k=5)

    if not docs:
        st.warning("No relevant documents found.")
    else:
        # Build context from retrieved docs
        context_parts = []
        for i, d in enumerate(docs):
            # doc might be langchain Document or dict
            if hasattr(d, "page_content"):
                content = d.page_content
                meta = getattr(d, "metadata", {}) or {}
            else:
                content = d.get("page_content") or d.get("content") or ""
                meta = d.get("metadata", {}) or {}

            source_name = meta.get("metadata_storage_name", meta.get("source", f"doc_{i}"))
            snippet = content.strip()
            context_parts.append(f"---\nSource: {source_name}\n{snippet}\n")

        context_combined = "\n".join(context_parts)

        # Construct the messages for Azure OpenAI chat completion (system + user)
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
                "If the answer is not contained in the context, say you don't know or ask for clarification."
            ),
        }

        user_message = {
            "role": "user",
            "content": f"Context:\n{context_combined}\n\nQuestion: {user_query}\n\nAnswer in concise form, and list which source(s) from the context you used."
        }

        # Call Azure OpenAI Chat Completion using openai SDK (model = deployment name)
        try:
            completion = openai.ChatCompletion.create(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[system_message, user_message],
                max_tokens=500,
                temperature=0,
            )
            answer = completion["choices"][0]["message"]["content"].strip()
        except Exception as e:
            st.error(f"Generation failed: {e}")
            answer = None

        if answer:
            st.subheader("📝 Answer")
            st.write(answer)

            st.subheader("📄 Source Chunks")
            for d in docs:
                if hasattr(d, "page_content"):
                    content = d.page_content
                    meta = getattr(d, "metadata", {}) or {}
                else:
                    content = d.get("page_content") or d.get("content") or ""
                    meta = d.get("metadata", {}) or {}

                source_name = meta.get("metadata_storage_name", meta.get("source", "unknown"))
                st.markdown(f"**Source:** {source_name}")
                st.write(content[:1000])
