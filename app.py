# app.py
import os
import streamlit as st
import tempfile
from typing import List, Any

# Azure clients
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient

# LangChain / community / helper libs (match your requirements)
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_core.documents import Document

# Try to import the new OpenAI v1 client; if not present, we'll fall back
openai_client_available = False
try:
    from openai import OpenAI as OpenAIClient
    openai_client_available = True
except Exception:
    openai_client_available = False

# Try LangChain AzureChatOpenAI (as fallback)
try:
    from langchain_openai import AzureChatOpenAI
    langchain_llm_available = True
except Exception:
    AzureChatOpenAI = None
    langchain_llm_available = False


# ---------------------------
# ----- Configuration -------
# ---------------------------
# Provide these as environment variables or via streamlit secrets
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or st.secrets.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") or st.secrets.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or st.secrets.get("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME") or st.secrets.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview") or st.secrets.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT") or st.secrets.get("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY") or st.secrets.get("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME") or st.secrets.get("AZURE_SEARCH_INDEX_NAME")

AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING") or st.secrets.get("AZURE_BLOB_CONNECTION_STRING")
AZURE_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER_NAME", "smartassistant-sops") or st.secrets.get("AZURE_BLOB_CONTAINER_NAME", "smartassistant-sops")


# ---------------------------
# ---- Simple Users (Opt A) -
# ---------------------------
# Replace / extend this with your real user store
USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "user1": {"password": "pass123", "role": "user"},
    "user2": {"password": "pass456", "role": "user"},
}


# ---------------------------
# ----- Utility helpers -----
# ---------------------------
def ensure_session_keys():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None
    if "users_data" not in st.session_state:
        st.session_state["users_data"] = {}  # per-user history etc.


def login_ui():
    st.sidebar.markdown("## 🔐 Login")
    user = st.sidebar.text_input("Username")
    pwd = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if user in USERS and USERS[user]["password"] == pwd:
            st.session_state["logged_in"] = True
            st.session_state["user_id"] = user
            if user not in st.session_state["users_data"]:
                st.session_state["users_data"][user] = {"history": []}
            st.sidebar.success(f"Signed in as {user}")
            st.experimental_rerun()
        else:
            st.sidebar.error("Invalid username or password")


def logout():
    st.session_state["logged_in"] = False
    st.session_state["user_id"] = None
    st.experimental_rerun()


def clear_user_ui_state():
    uid = st.session_state.get("user_id")
    if not uid:
        return
    st.session_state["users_data"].setdefault(uid, {})["last_query"] = None
    st.session_state["users_data"][uid]["last_answer"] = None
    st.session_state["users_data"][uid]["last_docs"] = []
    st.sidebar.success("Cleared your session view (UI only).")


# dedupe retrieved chunks by page_content (trimmed)
def dedupe_docs(docs: List[Any]) -> List[Any]:
    seen = set()
    unique = []
    for d in docs:
        text = ""
        if hasattr(d, "page_content"):
            text = d.page_content.strip()
        elif isinstance(d, dict):
            text = (d.get("page_content") or d.get("content") or "").strip()
        else:
            text = str(d).strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(d)
    return unique


# robust retriever call: try invoke -> get_relevant_documents -> get_relevant_sources
def call_retriever(retriever, query: str, k: int = 5):
    # try invoke (newer runnable-style retrievers)
    try:
        if hasattr(retriever, "invoke"):
            try:
                res = retriever.invoke(query)
            except TypeError:
                # some accept dict
                res = retriever.invoke({"query": query, "k": k})
            return list(res or [])[:k]
    except Exception:
        pass

    # try get_relevant_documents (older)
    try:
        if hasattr(retriever, "get_relevant_documents"):
            try:
                return retriever.get_relevant_documents(query)[:k]
            except TypeError:
                return retriever.get_relevant_documents(query, k=k)[:k]
    except Exception:
        pass

    # try get_relevant_sources
    try:
        if hasattr(retriever, "get_relevant_sources"):
            res = retriever.get_relevant_sources(query)
            return list(res or [])[:k]
    except Exception:
        pass

    raise RuntimeError("Retriever does not support known retrieval methods (invoke/get_relevant_documents/get_relevant_sources).")


# ---------------------------
# --- Validate env & clients
# ---------------------------
ensure_session_keys()

required_env = {
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_DEPLOYMENT": AZURE_OPENAI_DEPLOYMENT,
    "AZURE_EMBEDDING_DEPLOYMENT": AZURE_EMBEDDING_DEPLOYMENT,
    "AZURE_SEARCH_ENDPOINT": AZURE_SEARCH_ENDPOINT,
    "AZURE_SEARCH_ADMIN_KEY": AZURE_SEARCH_ADMIN_KEY,
    "AZURE_SEARCH_INDEX_NAME": AZURE_SEARCH_INDEX_NAME,
    "AZURE_BLOB_CONNECTION_STRING": AZURE_BLOB_CONNECTION_STRING,
}

missing = [k for k, v in required_env.items() if not v]
if missing:
    st.error(f"Missing required environment/secrets: {', '.join(missing)}")
    st.stop()

# create azure clients
try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
    blob_container_client = blob_service_client.get_container_client(AZURE_BLOB_CONTAINER)
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

# embeddings + vectorstore (LangChain wrapper)
embeddings = AzureOpenAIEmbeddings(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
    api_version=AZURE_OPENAI_API_VERSION,
)

vectorstore = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_ADMIN_KEY,
    index_name=AZURE_SEARCH_INDEX_NAME,
    embedding_function=embeddings.embed_query,
)


# ---------------------------
# --- LLM generation helper
# ---------------------------
# We'll try to use the new OpenAI client if available (openai>=1.0) with the chat completions API.
# If that fails or is not installed, we fallback to langchain_openai.AzureChatOpenAI.
def generate_answer_via_client(system_message: str, user_message: str, max_tokens: int = 500, temperature: float = 0.0):
    # preferred: OpenAI v1 client
    if openai_client_available:
        try:
            client = OpenAIClient(api_key=AZURE_OPENAI_API_KEY)
            # For Azure, some environments require base_url / api_type / api_version config.
            # If your Azure endpoint needs to be passed explicitly, uncomment and adjust:
            # client = OpenAIClient(api_key=AZURE_OPENAI_API_KEY, base_url=AZURE_OPENAI_ENDPOINT)
            resp = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            # fallthrough to langchain-based LLM
            last_err = e
    # fallback: LangChain AzureChatOpenAI
    if langchain_llm_available:
        try:
            llm = AzureChatOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                azure_deployment=AZURE_OPENAI_DEPLOYMENT,
                api_version=AZURE_OPENAI_API_VERSION,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Many LangChain LLMs implement .invoke or can be called directly; try safe options:
            if hasattr(llm, "invoke"):
                res = llm.invoke([{"role": "system", "content": system_message}, {"role": "user", "content": user_message}])
                # attempt to extract content robustly
                try:
                    return getattr(res, "content", None) or str(res)
                except Exception:
                    return str(res)
            else:
                # try call / generate interface
                output = llm.generate([{"role": "system", "content": system_message}, {"role": "user", "content": user_message}])
                # best-effort extract
                return str(output)
        except Exception as e2:
            last_err = e2
    # If both fail:
    raise RuntimeError(f"LLM generation failed (client+fallback). Last error: {locals().get('last_err', 'unknown')}")

# ---------------------------
# --- App UI + Logic
# ---------------------------
st.set_page_config(page_title="SmartAssistant (Role-based + Clear + Dedupe)", layout="wide")
st.title("🤖 SmartAssistantApp")

# login
if not st.session_state["logged_in"]:
    login_ui()
    st.stop()

# sidebar: user info / clear / logout
st.sidebar.markdown(f"**User:** {st.session_state['user_id']}")
if st.sidebar.button("Clear UI (only)"):
    clear_user_ui_state()
if st.sidebar.button("Logout"):
    logout()

# Main UI
st.header("Upload a document (PDF / TXT / DOCX)")
uploaded_file = st.file_uploader("Choose file", type=["pdf", "txt", "docx"])

if uploaded_file:
    # save to temp file for loader libs
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    # upload to blob (store original)
    try:
        blob_container_client.upload_blob(uploaded_file.name, uploaded_file, overwrite=True)
        st.success(f"Uploaded '{uploaded_file.name}' to blob storage.")
    except Exception as e:
        st.error(f"Failed to upload to blob: {e}")

    # load and chunk
    try:
        ext = uploaded_file.name.lower().split(".")[-1]
        if ext == "pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        else:
            # fallback to UnstructuredFileLoader
            loader = UnstructuredFileLoader(tmp_path)
            docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # create langchain_core Document objects with sanitized metadata
        doc_objs = []
        for c in chunks:
            meta = c.metadata or {}
            clean_meta = {
                "metadata_storage_name": meta.get("metadata_storage_name", uploaded_file.name),
                "page": int(meta.get("page", 0)) if meta.get("page") and str(meta.get("page")).isdigit() else 0,
                "source": str(meta.get("source", uploaded_file.name)),
            }
            doc_objs.append(Document(page_content=c.page_content, metadata=clean_meta))

        # index into Azure Search vector index
        try:
            vectorstore.add_documents(doc_objs)
            st.success(f"Indexed {len(doc_objs)} chunks into vector index.")
            # record in user history
            uid = st.session_state["user_id"]
            st.session_state["users_data"].setdefault(uid, {}).setdefault("history", []).append(f"Uploaded {uploaded_file.name}")
        except Exception as e:
            st.error(f"Indexing failed: {e}")

    except Exception as e:
        st.error(f"Failed to load or chunk file: {e}")

# Query area
st.markdown("---")
st.header("Ask a question (retrieval + generation)")

query = st.text_input("Your question:")

if st.button("Clear last query and results"):
    # clear user UI state
    clear_user_ui_state()

if query:
    st.info("Retrieving relevant chunks...")
    try:
        retr = vectorstore.as_retriever(search_kwargs={"k": 8})
    except Exception:
        retr = vectorstore.as_retriever()

    try:
        docs = call_retriever(retr, query, k=8)
    except Exception as e:
        st.error(f"Retrieval failed: {e}")
        docs = []

    if not docs:
        st.warning("No relevant documents found.")
    else:
        # dedupe and reduce
        docs = dedupe_docs(docs)
        docs = docs[:3]

        # build context
        context_parts = []
        for i, d in enumerate(docs, 1):
            if hasattr(d, "page_content"):
                text = d.page_content
                meta = getattr(d, "metadata", {}) or {}
            else:
                text = d.get("page_content") or d.get("content") or ""
                meta = d.get("metadata", {}) or {}

            src = meta.get("metadata_storage_name") or meta.get("source") or f"doc_{i}"
            context_parts.append(f"Source: {src}\n\n{text}")

        context_combined = "\n\n---\n\n".join(context_parts)

        # prepare system/user messages
        system_message = "You are an assistant. Use ONLY the provided context to answer. If not present, say 'Information not available in the provided documents.'"
        user_message = f"Context:\n{context_




