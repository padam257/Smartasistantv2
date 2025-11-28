# app.py
import os
import streamlit as st
import tempfile
from typing import List, Dict

# OpenAI new client (>=1.0.0)
from openai import OpenAI

# Azure clients
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient

# LangChain / community wrappers for embeddings & vectorstore
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch

# Document loader + splitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document model (try core path, fallback to older schema)
try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document


# ---------------------------------------------------------
#  CONFIG: environment variables / streamlit secrets
# ---------------------------------------------------------
# You may put these in Streamlit secrets or environment variables.
# Example Streamlit secrets keys used below:
# - AZURE_OPENAI_ENDPOINT
# - AZURE_OPENAI_API_KEY
# - AZURE_OPENAI_API_VERSION
# - AZURE_OPENAI_DEPLOYMENT_NAME
# - AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
# - AZURE_SEARCH_ENDPOINT
# - AZURE_SEARCH_ADMIN_KEY
# - AZURE_SEARCH_INDEX_NAME
# - AZURE_BLOB_CONNECTION_STRING
# - AZURE_BLOB_CONTAINER_NAME

# Read from st.secrets if present, otherwise os.environ
def get_secret(k: str):
    return st.secrets.get(k) if hasattr(st, "secrets") and k in st.secrets else os.environ.get(k)

AZURE_OPENAI_ENDPOINT = get_secret("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = get_secret("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = get_secret("AZURE_OPENAI_API_VERSION") or "2024-02-15-preview"
AZURE_OPENAI_DEPLOYMENT_NAME = get_secret("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = get_secret("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

AZURE_SEARCH_ENDPOINT = get_secret("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = get_secret("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME = get_secret("AZURE_SEARCH_INDEX_NAME")

AZURE_BLOB_CONNECTION_STRING = get_secret("AZURE_BLOB_CONNECTION_STRING")
AZURE_BLOB_CONTAINER_NAME = get_secret("AZURE_BLOB_CONTAINER_NAME") or "smartassistant-sops"

# Minimal required check
required = {
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_OPENAI_DEPLOYMENT_NAME": AZURE_OPENAI_DEPLOYMENT_NAME,
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    "AZURE_SEARCH_ENDPOINT": AZURE_SEARCH_ENDPOINT,
    "AZURE_SEARCH_ADMIN_KEY": AZURE_SEARCH_ADMIN_KEY,
    "AZURE_SEARCH_INDEX_NAME": AZURE_SEARCH_INDEX_NAME,
    "AZURE_BLOB_CONNECTION_STRING": AZURE_BLOB_CONNECTION_STRING,
}

missing = [k for k, v in required.items() if not v]
if missing:
    st.error("Missing required configuration: " + ", ".join(missing))
    st.stop()


# ---------------------------------------------------------
# Simple Role-Based (Option A) credentials
# Replace / integrate with your user DB if needed
# ---------------------------------------------------------
USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "user1": {"password": "pass123", "role": "user"},
    "user2": {"password": "pass456", "role": "user"},
}

# Initialize session-state containers
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "users" not in st.session_state:
    st.session_state["users"] = {}  # maps username -> {history: [...], uploads: [...]}
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""
if "last_docs" not in st.session_state:
    st.session_state["last_docs"] = []


def login_box():
    st.sidebar.subheader("🔐 Sign in")
    username = st.sidebar.text_input("Username", key="login_username")
    pwd = st.sidebar.text_input("Password", type="password", key="login_pwd")
    if st.sidebar.button("Sign in"):
        if username in USERS and USERS[username]["password"] == pwd:
            st.session_state["logged_in"] = True
            st.session_state["user_id"] = username
            st.session_state["role"] = USERS[username]["role"]
            if username not in st.session_state["users"]:
                st.session_state["users"][username] = {"history": [], "uploads": []}
            st.experimental_rerun()
        else:
            st.sidebar.error("Invalid username or password")


def logout():
    keys = ["logged_in", "user_id", "role", "last_answer", "last_docs"]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()


def clear_user_ui():
    """Clear only UI/session history for current user (no deletion from Azure Search)"""
    user = st.session_state.get("user_id")
    if user and user in st.session_state["users"]:
        st.session_state["users"][user]["history"] = []
        st.session_state["last_answer"] = ""
        st.session_state["last_docs"] = []
    st.success("Cleared session view (user-local).")


# ---------------------------------------------------------
# Initialize Azure clients (blob + search wrapper + openai client)
# ---------------------------------------------------------
try:
    blob_service = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
    blob_container_client = blob_service.get_container_client(AZURE_BLOB_CONTAINER_NAME)
except Exception as e:
    st.error(f"Failed to initialize Blob client: {e}")
    st.stop()

# Note: we use the langchain wrapper AzureSearch; below we build embeddings that it uses.
embeddings = AzureOpenAIEmbeddings(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    api_version=AZURE_OPENAI_API_VERSION,
)

# New OpenAI client for chat (openai>=1.x)
# Configure the OpenAI client to talk to Azure
openai_client = OpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    base_url=AZURE_OPENAI_ENDPOINT,
    api_type="azure",
    api_version=AZURE_OPENAI_API_VERSION
)


# ---------------------------------------------------------
# Vectorstore factory
# ---------------------------------------------------------
def get_vectorstore() -> AzureSearch:
    vs = AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_ADMIN_KEY,
        index_name=AZURE_SEARCH_INDEX_NAME,
        embedding_function=embeddings.embed_query,
    )
    return vs


# ---------------------------------------------------------
# Utilities: loading, chunking, creating Document objects
# ---------------------------------------------------------
def load_file_to_chunks(file_obj, filename: str):
    """
    Load a file-like object to langchain Document chunks (langchain Document instances)
    """
    ext = filename.lower().split(".")[-1]
    tmp_path = None
    # Some loaders accept a path, so write file to a temp path first
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(file_obj.getbuffer())
        tmp_path = tmp.name

    if ext == "pdf":
        loader = PyPDFLoader(tmp_path)
    elif ext in ("txt", "text"):
        loader = TextLoader(tmp_path)
    elif ext in ("docx", "doc"):
        loader = UnstructuredFileLoader(tmp_path)
    else:
        raise ValueError("Unsupported file type")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # sanitize metadata and ensure Document objects
    out_chunks: List[Document] = []
    for c in chunks:
        meta = c.metadata or {}
        clean_meta = {
            "metadata_storage_name": meta.get("metadata_storage_name", filename),
            "page": int(meta.get("page", 0)) if meta.get("page", 0) is not None else 0,
            "source": meta.get("source", filename),
        }
        doc_obj = Document(page_content=c.page_content, metadata=clean_meta)
        out_chunks.append(doc_obj)

    return out_chunks


# ---------------------------------------------------------
# Dedupe function to remove near-exact duplicate chunks (by text)
# ---------------------------------------------------------
def dedupe_docs(docs: List) -> List:
    seen = set()
    unique = []
    for d in docs:
        text = d.page_content.strip() if hasattr(d, "page_content") else (d.get("page_content") or "").strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(d)
    return unique


# ---------------------------------------------------------
# Robust retriever invocation (supports different retriever APIs)
# ---------------------------------------------------------
def call_retriever(retriever, query: str, k: int = 5):
    # attempt invoke (new style)
    try:
        if hasattr(retriever, "invoke"):
            try:
                res = retriever.invoke({"query": query, "k": k})
            except TypeError:
                # some retrievers accept plain string
                res = retriever.invoke(query)
            return list(res or [])[:k]
    except Exception:
        pass

    # try older api
    try:
        if hasattr(retriever, "get_relevant_documents"):
            try:
                return retriever.get_relevant_documents(query)[:k]
            except TypeError:
                return retriever.get_relevant_documents(query, k=k)[:k]
    except Exception:
        pass

    raise RuntimeError("Retriever does not support known methods (invoke/get_relevant_documents).")


# ---------------------------------------------------------
# Upload -> index pipeline
# ---------------------------------------------------------
def upload_file_and_index(file_obj):
    filename = file_obj.name
    # upload to blob storage
    try:
        blob_container_client.upload_blob(name=filename, data=file_obj, overwrite=True)
    except Exception as e:
        st.error(f"Failed to upload to blob: {e}")
        return False

    # create chunks
    try:
        chunks = load_file_to_chunks(file_obj, filename)
    except Exception as e:
        st.error(f"Failed to load/chunk file: {e}")
        return False

    # index into vectorstore (Document objects required)
    try:
        vs = get_vectorstore()
        vs.add_documents(chunks)
    except Exception as e:
        st.error(f"Indexing failed: {e}")
        return False

    # store metadata in session user history
    user = st.session_state.get("user_id")
    if user:
        st.session_state["users"].setdefault(user, {"history": [], "uploads": []})
        st.session_state["users"][user]["uploads"].append(filename)
        st.session_state["users"][user]["history"].append(f"Uploaded {filename}")

    return True


# ---------------------------------------------------------
# Generation: build context from retrieved docs and call Azure OpenAI (OpenAI client 1.x)
# ---------------------------------------------------------
def generate_answer_from_docs(question: str, docs: List[Document]) -> str:
    # build context
    combined = ""
    for i, d in enumerate(docs, start=1):
        # include source short metadata
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("metadata_storage_name", meta.get("source", f"doc_{i}"))
        combined += f"---\nSource: {src}\n{d.page_content}\n\n"

    system = {
        "role": "system",
        "content": "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
                   "If the answer is not contained in the context, say you don't know."
    }
    user = {
        "role": "user",
        "content": f"Context:\n{combined}\n\nQuestion: {question}\n\nAnswer concisely and cite sources from the context."
    }

    try:
        resp = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[system, user],
            max_tokens=500,
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}")


# ---------------------------------------------------------
# UI layout
# ---------------------------------------------------------
st.set_page_config(page_title="SmartAssistantApp", layout="wide")
st.title("🤖 SmartAssistantApp (Role-based, per-user sessions)")

# authentication
if not st.session_state.get("logged_in", False):
    login_box()
    st.stop()

# logged-in UI
user = st.session_state["user_id"]
role = st.session_state.get("role", "user")

with st.sidebar:
    st.write(f"Signed in as: **{user}** ({role})")
    if st.button("Logout"):
        logout()
    if st.button("Clear my UI session"):
        clear_user_ui()

# Upload area
st.header("Upload document (PDF / TXT / DOCX)")
uploaded = st.file_uploader("Select file", type=["pdf", "txt", "docx"])
if uploaded:
    st.info(f"Uploading & indexing {uploaded.name} ...")
    ok = upload_file_and_index(uploaded)
    if ok:
        st.success("Upload + index complete.")
    else:
        st.error("Upload/index failed (see messages above).")

# Query area
st.header("Ask a question")
question = st.text_input("Enter your question here:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        # build retriever and call robust helper
        try:
            vs = get_vectorstore()
            retriever = vs.as_retriever(search_kwargs={"k": 8})
            docs = call_retriever(retriever, question, k=8)
            docs = dedupe_docs(docs)
            docs = docs[:3]  # keep top 3 unique chunks
        except Exception as e:
            st.error(f"Retrieval failed: {e}")
            docs = []

        if not docs:
            st.warning("No relevant documents found.")
        else:
            try:
                answer = generate_answer_from_docs(question, docs)
                st.subheader("Answer")
                st.write(answer)
                st.subheader("Sources")
                for i, d in enumerate(docs, 1):
                    meta = getattr(d, "metadata", {}) or {}
                    src = meta.get("metadata_storage_name", meta.get("source", f"doc_{i}"))
                    st.markdown(f"**Source {i}:** {src}")
                    st.write(d.page_content[:1000])
                    st.markdown("---")

                # store history for this user
                st.session_state["users"].setdefault(user, {"history": [], "uploads": []})
                st.session_state["users"][user]["history"].append(f"Q: {question} -> A preview: {answer[:120]}...")
                st.session_state["last_answer"] = answer
                st.session_state["last_docs"] = docs
            except Exception as e:
                st.error(f"Generation failed: {e}")


# Show user history (private)
st.header("Your session history (private)")
for item in st.session_state["users"].get(user, {}).get("history", []):
    st.write("- ", item)
