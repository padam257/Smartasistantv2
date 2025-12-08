# app.py - Final production-ready SmartAssistant
# Features:
# - Multi-user auth from environment variables (USER_1..USER_24)
# - Roles: admin (full), editor (upload+index), reader (query-only)
# - Per-user conversation history in session_state
# - Clear conversation button
# - Upload PDF -> extract (pypdf) -> chunk -> embed -> index to Azure Cognitive Search
# - Retrieval using VectorQuery(kind="vector") against content_vector field
# - Uses Azure OpenAI embeddings (deployment set in AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
# - Uses Azure OpenAI chat deployment (AZURE_OPENAI_DEPLOYMENT_NAME) for generation

# app.py
"""
Final production-ready SmartAssistant (Streamlit)
- Matches Azure Search index fields exactly:
    id (String key)
    content (String)
    content_vector (SingleCollection, 1536)
    metadata (String)
    page (Int32)

- Requirements (env/App Service settings):
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_DEPLOYMENT_NAME           -> gpt-4o (chat/completion deployment)
    AZURE_OPENAI_API_VERSION               -> e.g. 2024-02-15-preview
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT      -> text-embedding-ada-002 (embedding deployment)
    AZURE_SEARCH_ENDPOINT
    AZURE_SEARCH_KEY
    AZURE_SEARCH_INDEX
    AZURE_STORAGE_CONNECTION_STRING
    AZURE_STORAGE_CONTAINER
    USER_1_USERNAME / USER_1_PASSWORD / USER_1_ROLE ... (up to USER_24_*)
"""

import os
import uuid
import tempfile
import streamlit as st
from typing import List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv

# Azure SDK
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorQuery

# OpenAI (Azure) client
from openai import AzureOpenAI

# LangChain wrappers
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF text extraction
from pypdf import PdfReader

# Optional: support Document type from langchain if available
try:
    from langchain_core.documents import Document as LC_Document
except Exception:
    try:
        from langchain.schema import Document as LC_Document
    except Exception:
        LC_Document = None  # fallback to dicts

# Load local .env for local dev
load_dotenv()

# -------------------------
# helper to read secrets (Streamlit secrets or env)
# -------------------------
def get_secret(key: str) -> str:
    try:
        if hasattr(st, "secrets") and st.secrets and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key)


# -------------------------
# Required environment variables (validate early)
# -------------------------
st.set_page_config(page_title="SmartAssistant", layout="wide")

AZURE_OPENAI_ENDPOINT = get_secret("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = get_secret("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = get_secret("AZURE_OPENAI_DEPLOYMENT_NAME")  # gpt-4o (chat)
AZURE_OPENAI_API_VERSION = get_secret("AZURE_OPENAI_API_VERSION") or get_secret("OPENAI_API_VERSION")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = get_secret("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")  # text-embedding-ada-002

AZURE_SEARCH_ENDPOINT = get_secret("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = get_secret("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = get_secret("AZURE_SEARCH_INDEX")

AZURE_STORAGE_CONNECTION = get_secret("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = get_secret("AZURE_STORAGE_CONTAINER")

required_map = {
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_OPENAI_DEPLOYMENT_NAME": AZURE_OPENAI_DEPLOYMENT_NAME,
    "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    "AZURE_SEARCH_ENDPOINT": AZURE_SEARCH_ENDPOINT,
    "AZURE_SEARCH_KEY": AZURE_SEARCH_KEY,
    "AZURE_SEARCH_INDEX": AZURE_SEARCH_INDEX,
    "AZURE_STORAGE_CONNECTION_STRING": AZURE_STORAGE_CONNECTION,
    "AZURE_STORAGE_CONTAINER": AZURE_STORAGE_CONTAINER,
}

_missing = [k for k, v in required_map.items() if not v]
if _missing:
    st.error("Missing required environment variables: " + ", ".join(_missing))
    st.stop()

# -------------------------
# Multi-user loader from env (USER_1..USER_24)
# -------------------------
def load_users() -> Dict[str, Dict[str, str]]:
    users = {}
    for i in range(1, 25):
        uname = get_secret(f"USER_{i}_USERNAME")
        pwd = get_secret(f"USER_{i}_PASSWORD")
        role = get_secret(f"USER_{i}_ROLE")
        if uname and pwd:
            users[uname] = {"password": pwd, "role": role or "reader"}
    return users


# -------------------------
# Session init
# -------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "users_history" not in st.session_state:
    st.session_state["users_history"] = {}

# -------------------------
# Login UI
# -------------------------
def login():
    st.title("🔐 SmartAssistant Login")
    users = load_users()
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        if username in users and password == users[username]["password"]:
            st.session_state["authenticated"] = True
            st.session_state["user"] = username
            st.session_state["role"] = users[username]["role"]
            st.session_state["users_history"].setdefault(username, {"history": []})
            st.success(f"Welcome, {username} ({st.session_state['role']})")
            st.rerun()
        else:
            st.error("Invalid username or password")


if not st.session_state["authenticated"]:
    login()
    st.stop()

USER = st.session_state["user"]
ROLE = st.session_state.get("role", "reader")

# -------------------------
# Initialize clients (OpenAI embedding + chat, Search, Blob)
# -------------------------
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)

embedder = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    api_version=AZURE_OPENAI_API_VERSION,
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY),
)

blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION)
blob_container = blob_service.get_container_client(AZURE_STORAGE_CONTAINER)

# -------------------------
# PDF extraction using pypdf
# -------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> List[str]:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        tmp_path = tmp.name

    reader = PdfReader(tmp_path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return pages

# -------------------------
# Chunking pages into documents
# -------------------------
def chunk_pages_into_documents(pages: List[str], source_name: str, chunk_size: int = 1200, chunk_overlap: int = 50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []
    for page_num, page_text in enumerate(pages, start=1):
        if LC_Document:
            base = LC_Document(page_content=page_text, metadata={"metadata": source_name, "page": page_num})
        else:
            base = {"page_content": page_text, "metadata": {"metadata": source_name, "page": page_num}}
        split_docs = splitter.split_documents([base])
        for sd in split_docs:
            md = sd.metadata or {}
            md["metadata"] = md.get("metadata", source_name)
            sd.metadata = md
            docs.append(sd)
    return docs

# -------------------------
# Indexing: build documents with EXACT fields for your index
# Fields: id, content, content_vector, metadata, page
# -------------------------
def index_documents_to_azure_search(docs) -> (bool, Any):
    if not docs:
        return True, "No docs"

    texts = [d.page_content for d in docs]
    try:
        vectors = embedder.embed_documents(texts)
    except Exception as e:
        return False, f"Embedding failed: {e}"

    azure_docs = []
    for i, d in enumerate(docs):
        md = d.metadata or {}
        doc_id = str(uuid.uuid4())
        azure_docs.append({
            "id": doc_id,
            "content": d.page_content,
            "content_vector": vectors[i],         # MUST match index field
            "metadata": md.get("metadata", "unknown"),
            "page": int(md.get("page", 0) or 0),
        })

    try:
        result = search_client.upload_documents(documents=azure_docs)
        return True, result
    except Exception as e:
        return False, f"Search upload failed: {e}"

# -------------------------
# Upload + index pipeline
# -------------------------
def upload_and_index_pdf(uploaded_file) -> bool:
    try:
        file_bytes = uploaded_file.getvalue()
    except Exception:
        file_bytes = uploaded_file.read()

    # upload original PDF to blob
    try:
        blob_name = f"{uuid.uuid4()}-{uploaded_file.name}"
        blob_container.upload_blob(blob_name, file_bytes, overwrite=True)
    except Exception as e:
        st.error(f"Blob upload failed: {e}")
        return False

    # extract text pages
    try:
        pages = extract_text_from_pdf_bytes(file_bytes)
    except Exception as e:
        st.error(f"PDF extraction failed: {e}")
        return False

    # chunk pages into docs
    try:
        docs = chunk_pages_into_documents(pages, source_name=uploaded_file.name)
    except Exception as e:
        st.error(f"Chunking failed: {e}")
        return False

    # index
    ok, info = index_documents_to_azure_search(docs)
    if not ok:
        st.error(f"Indexing failed: {info}")
        return False

    return True

# -------------------------
def retrieve_top_chunks(query: str, k: int = 5):
    """
    Returns list[dict] with id, content, metadata, page.
    Uses Azure OpenAI embeddings + Azure Search vector search.
    Compatible with all stable SDK versions.
    """

    # ----- 1) Embed query -----
    try:
        try:
            qvec = embedder.embed_query(query)
        except:
            resp = openai_client.embeddings.create(
                model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                input=query
            )
            qvec = resp["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        return []

    # ----- 2) Search (legacy syntax, fully compatible) -----
    try:
        results = search_client.search(
            search_text=None,
            vectors=[{
                "value": qvec,
                "fields": "content_vector",
                "k": k
            }],
            select=["id", "content", "metadata", "page"],
            top=k
        )
    except Exception as e:
        st.error(f"Azure Search query failed: {e}")
        return []

    # ----- 3) Format results -----
    docs = []
    try:
        for r in results:
            docs.append({
                "id": r.get("id"),
                "content": r.get("content"),
                "source": r.get("metadata"),
                "page": r.get("page")
            })
    except Exception as e:
        st.error(f"Failed while processing search results: {e}")
        return []
    return docs


# -------------------------
# Deduplicate exact duplicate chunks
# -------------------------
def dedupe_chunks(chunks: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for c in chunks:
        txt = (c.get("content") or "").strip()
        if not txt:
            continue
        if txt not in seen:
            seen.add(txt)
            out.append(c)
    return out

# -------------------------
# Generation: Azure OpenAI chat
# -------------------------
def generate_answer(question: str, docs: List[Dict]) -> str:
    context = "\n\n---\n\n".join([f"Source: {d['source']} (page {d.get('page')})\n{d['content']}" for d in docs[:3]])
    system = {"role": "system", "content": "You are a helpful assistant. Answer using ONLY the provided context. If not present, say you don't know."}
    user_msg = {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    try:
        resp = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[system, user_msg],
            max_tokens=500,
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Generation failed: {e}")
        return ""

# -------------------------
# UI
# -------------------------
st.title("🤖 SmartAssistant — RAG on Azure Search")
st.caption(f"Signed in as: **{USER}** (role: {ROLE})")

# Clear conversation
if st.button("🧹 Clear Conversation"):
    st.session_state["users_history"][USER]["history"] = []
    st.success("Cleared conversation (UI only)")
    st.rerun()

# Upload (admin + editor)
if ROLE in ("admin", "editor"):
    uploaded = st.file_uploader("Upload PDF to index", type=["pdf"])
    if uploaded:
        st.info(f"Uploading & indexing {uploaded.name} ...")
        ok = upload_and_index_pdf(uploaded)
        if ok:
            st.success("Upload + indexing completed")
            st.session_state["users_history"][USER]["history"].append(f"Uploaded {uploaded.name}")
else:
    st.info("Upload disabled for reader role")

st.divider()

# Query
query = st.text_input("Ask a question about indexed documents:")
if st.button("Run Query"):
    if not query.strip():
        st.warning("Please enter a question")
    else:
        results = retrieve_top_chunks(query, k=8)
        results = dedupe_chunks(results)
        results = results[:3]
        if not results:
            st.warning("No relevant documents found.")
        else:
            answer = generate_answer(query, results)
            st.subheader("📝 Answer")
            st.write(answer)
            st.subheader("📄 Source Chunks")
            for i, d in enumerate(results, start=1):
                st.markdown(f"**Source #{i}:** {d['source']} (page {d.get('page')})")
                st.write(d["content"][:1500])
                st.markdown("---")
            st.session_state["users_history"][USER]["history"].append(f"Q: {query} -> A preview: {answer[:120]}")

# Show per-user history
st.subheader("Your session history (private)")
history = st.session_state["users_history"].get(USER, {}).get("history", [])
for h in history:
    st.write("- ", h)

# EOF







