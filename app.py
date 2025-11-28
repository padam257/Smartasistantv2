# app.py
import os
import openai
import streamlit as st
from pathlib import Path
import tempfile
from typing import List

from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from langchain_openai import AzureChatOpenAI

# embeddings + vectorstore wrapper
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch

# text splitters & loaders
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader

# Document class for indexing
from langchain_core.documents import Document

# -------------------------
# Configuration (env vars)
# -------------------------
openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  # chat / completion deployment
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

# -------------------------
# Helper: robust retriever invocation
# -------------------------
def call_retriever(retriever, query: str, k: int = 5):
    """
    Call a retriever object that might implement different APIs:
      - invoke / ainvoke (new Runnable-style)
      - get_relevant_documents (older style)
      - get_relevant_documents(query, k=k)
    Returns a list of Document-like objects or dicts.
    """
    # Try invoke with simple string
    try:
        if hasattr(retriever, "invoke"):
            # Some retrievers accept a plain string
            try:
                res = retriever.invoke(query)
            except TypeError:
                # Some expect a dict: {"query": query, "k": k}
                res = retriever.invoke({"query": query, "k": k})
            return list(res or [])[:k]
    except Exception:
        # fallthrough to next attempt
        pass

    # Try get_relevant_documents
    try:
        if hasattr(retriever, "get_relevant_documents"):
            try:
                return retriever.get_relevant_documents(query)[:k]
            except TypeError:
                return retriever.get_relevant_documents(query, k=k)[:k]
    except Exception:
        pass

    # Try get_relevant_sources or similar
    try:
        if hasattr(retriever, "get_relevant_sources"):
            return retriever.get_relevant_sources(query)[:k]
    except Exception:
        pass

    # Nothing worked
    raise RuntimeError("Retriever does not support known retrieval methods (invoke/get_relevant_documents).")

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="SmartAssistantApp", layout="wide")
st.title("🤖 SmartAssistantApp: SOP GenAI")
st.markdown(
    "Upload SOPs (pdf/txt/docx), index them, and ask questions. Answers are produced using Azure OpenAI and Azure Cognitive Search vectors."
)

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
        to_index: List[Document] = []
        for c in chunks:
            meta = c.metadata or {}
            metadata_storage_name = meta.get("metadata_storage_name", file_name)
            page = meta.get("page", 0)
            source = meta.get("source", metadata_storage_name)

            clean_meta = {
                "metadata_storage_name": metadata_storage_name,
                "page": int(page) if isinstance(page, (int, str)) and str(page).isdigit() else 0,
                "source": str(source),
            }

            # Create a langchain_core Document so AzureSearch wrapper sees .id etc.
            doc_obj = Document(page_content=c.page_content, metadata=clean_meta)
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
    # Build retriever (with optional filter)
    try:
        if query_scope != "All Documents":
            retriever = vectorstore.as_retriever(search_kwargs={"filter": f"metadata_storage_name eq '{query_scope}'"})
        else:
            retriever = vectorstore.as_retriever()
    except Exception as e:
        st.error(f"Failed to create retriever: {e}")
        st.stop()

    # Retrieve docs using robust helper
    try:
        docs = call_retriever(retriever, user_query, k=5)
    except Exception as e:
        st.error(f"Retrieval failed: {e}")
        docs = []

    if not docs:
        st.warning("No relevant documents found.")
    else:
        # Build context from retrieved docs
        context_parts = []
        normalized_docs = []
        for i, d in enumerate(docs):
            # doc might be langchain Document or dict
            if hasattr(d, "page_content"):
                content = d.page_content
                meta = getattr(d, "metadata", {}) or {}
            else:
                # Some retrievers return dict-like objects
                content = d.get("page_content") or d.get("content") or d.get("text") or ""
                meta = d.get("metadata", {}) or {}

            source_name = meta.get("metadata_storage_name", meta.get("source", f"doc_{i}"))
            snippet = content.strip()
            context_parts.append(f"---\nSource: {source_name}\n{snippet}\n")

            # normalize into a simple dict for display / debugging
            normalized_docs.append({"source": source_name, "content": snippet})

        context_combined = "\n".join(context_parts)

        # Construct messages for Azure OpenAI chat
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
        llm = AzureChatOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            temperature=0,
            max_tokens=500,
        )

        try:
            response = llm.invoke([system_message, user_message])
            answer = response.content
        except Exception as e:
            st.error(f"Generation failed: {e}")
            answer = None

        if answer:
            st.subheader("📝 Answer")
            st.write(answer)

            st.subheader("📄 Source Chunks")
            for nd in normalized_docs:
                st.markdown(f"**Source:** {nd['source']}")
                st.write(nd["content"][:1000])

