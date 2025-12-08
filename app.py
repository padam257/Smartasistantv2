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

# app.py - SmartAssistant (REST vector search, no LangChain retrieval)
import os
import uuid
import json
import tempfile
import requests
import streamlit as st
from typing import List, Dict, Any
from datetime import datetime
from pypdf import PdfReader
from dotenv import load_dotenv

# OpenAI Azure client (wrap)
from openai import AzureOpenAI

# Load .env locally if present
load_dotenv()

# -------------------------
# Helper: get secret (Streamlit secrets or env)
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
AZURE_OPENAI_DEPLOYMENT_NAME = get_secret("AZURE_OPENAI_DEPLOYMENT_NAME")  # chat deployment (gpt-4o)
AZURE_OPENAI_API_VERSION = get_secret("AZURE_OPENAI_API_VERSION") or get_secret("OPENAI_API_VERSION")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = get_secret("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")  # text-embedding-ada-002

AZURE_SEARCH_ENDPOINT = get_secret("AZURE_SEARCH_ENDPOINT")  # e.g. https://<service>.search.windows.net
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
# Initialize Azure OpenAI client (Azure wrapper)
# -------------------------
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)

# -------------------------
# Simple text splitter (no LangChain)
# -------------------------
def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return [c for c in chunks if c]

# -------------------------
# PDF extraction
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
# Azure Search REST helpers
# -------------------------
API_VERSION = "2025-09-01"

def docs_index_url() -> str:
    return f"{AZURE_SEARCH_ENDPOINT.rstrip('/')}/indexes/{AZURE_SEARCH_INDEX}/docs/index?api-version={API_VERSION}"

def docs_search_url() -> str:
    return f"{AZURE_SEARCH_ENDPOINT.rstrip('/')}/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version={API_VERSION}"

def docs_search_all_url() -> str:
    return docs_search_url()

def headers_for_search() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_KEY
    }

# Upload documents (bulk). Each doc must contain only fields present in index:
# id, content, content_vector, metadata, page
def rest_index_documents(azure_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = {"value": []}
    for d in azure_docs:
        item = d.copy()
        item["@search.action"] = "upload"  # upload new or replace
        payload["value"].append(item)
    resp = requests.post(docs_index_url(), headers=headers_for_search(), json=payload, timeout=60)
    if resp.status_code >= 300:
        raise Exception(f"Index REST error {resp.status_code}: {resp.text}")
    return resp.json()

# Vector search via REST with vectorQueries (kind="vector")
def rest_vector_search(query_vector: List[float], k: int = 5, filter_metadata: List[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "top": k,
        "select": ["id", "content", "metadata", "page"],
        "vectorQueries": [
            {
                "kind": "vector",
                "vector": query_vector,
                "fields": "content_vector",
                "k": k
            }
        ]
    }
    # optional filter: metadata in ('a','b') -> build an OData-in expression
    if filter_metadata:
        # build list of quoted values
        quoted = ",".join([json.dumps(x) for x in filter_metadata])
        # metadata is string field - use 'metadata eq 'value'' or use 'metadata in ( ... )' — the service supports 'metadata in (..)'
        payload["filter"] = f"metadata in ({quoted})"
    resp = requests.post(docs_search_url(), headers=headers_for_search(), json=payload, timeout=60)
    if resp.status_code >= 300:
        raise Exception(f"Search REST error {resp.status_code}: {resp.text}")
    return resp.json()

# Fetch list of existing metadata values (filenames) from index for selection
def rest_get_unique_metadata(top: int = 1000) -> List[str]:
    # Get up to `top` docs and extract metadata
    payload = {
        "top": top,
        "select": ["metadata"]
    }
    resp = requests.post(docs_search_all_url(), headers=headers_for_search(), json=payload, timeout=30)
    if resp.status_code != 200:
        return []
    data = resp.json()
    values = [item.get("metadata") for item in data.get("value", []) if item.get("metadata")]
    # unique preserve order
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

# -------------------------
# Embedding helpers (Azure OpenAI)
# -------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    # batch embed using AzureOpenAI client
    # openai_client.embeddings.create accepts model and input list
    if not texts:
        return []
    try:
        resp = openai_client.embeddings.create(
            model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            input=texts
        )
        # resp["data"] is list of {embedding: [...]}
        return [d["embedding"] for d in resp["data"]]
    except Exception as e:
        raise Exception(f"Embedding API failed: {e}")

def embed_text(text: str) -> List[float]:
    return embed_texts([text])[0]

# -------------------------
# Upload / indexing pipeline
# -------------------------
def upload_and_index_pdf(uploaded_file, chunk_size=1000, chunk_overlap=200) -> bool:
    """Upload PDF to blob, extract, chunk, embed (batch), and index via REST"""
    # read bytes
    try:
        file_bytes = uploaded_file.getvalue()
    except Exception:
        file_bytes = uploaded_file.read()

    # upload to blob storage (store original)
    try:
        from azure.storage.blob import BlobServiceClient
        blob_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION)
        container = blob_client.get_container_client(AZURE_STORAGE_CONTAINER)
        blob_name = f"{uuid.uuid4()}-{uploaded_file.name}"
        container.upload_blob(blob_name, file_bytes, overwrite=True)
    except Exception as e:
        st.error(f"Blob upload failed: {e}")
        return False

    # extract text pages
    try:
        pages = extract_text_from_pdf_bytes(file_bytes)
    except Exception as e:
        st.error(f"PDF extraction failed: {e}")
        return False

    # create chunks per page
    chunk_objs = []  # list of dicts: {"content":..., "page": int}
    for page_num, page_text in enumerate(pages, start=1):
        page_chunks = split_text(page_text or "", chunk_size=chunk_size, overlap=chunk_overlap)
        for pc in page_chunks:
            chunk_objs.append({"content": pc, "page": page_num})

    if not chunk_objs:
        st.warning("No extractable text found in PDF.")
        return False

    # embed all chunk texts in batches
    texts = [c["content"] for c in chunk_objs]
    try:
        vectors = embed_texts(texts)
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        return False

    # prepare azure docs matching index fields only
    azure_docs = []
    for idx, c in enumerate(chunk_objs):
        azure_docs.append({
            "id": str(uuid.uuid4()),
            "content": c["content"],
            "content_vector": vectors[idx],
            "metadata": uploaded_file.name,
            "page": int(c["page"] or 0)
        })

    # index via REST
    try:
        resp = rest_index_documents(azure_docs)
    except Exception as e:
        st.error(f"Indexing failed: {e}")
        return False

    return True

# -------------------------
# Retrieval pipeline
# -------------------------
def retrieve_top_chunks(query: str, k: int = 5, selected_docs: List[str] = None) -> List[Dict[str, Any]]:
    # embed query
    try:
        qvec = embed_text(query)
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        return []

    try:
        resp = rest_vector_search(qvec, k=k, filter_metadata=selected_docs)
    except Exception as e:
        st.error(f"Vector search failed: {e}")
        return []

    values = resp.get("value", [])
    docs = []
    for item in values:
        docs.append({
            "id": item.get("id"),
            "content": item.get("content"),
            "metadata": item.get("metadata"),
            "page": item.get("page")
        })
    return docs

# -------------------------
# Dedupe exact duplicates
# -------------------------
def dedupe_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
# Generation using Azure OpenAI chat (chat deployment)
# -------------------------
def generate_answer(question: str, chunks: List[Dict[str, Any]]) -> str:
    # Build context from up to top 3 chunks
    context = "\n\n---\n\n".join([f"Source: {c.get('metadata','unknown')} (page {c.get('page')})\n{c.get('content')}" for c in chunks[:3]])
    system = {"role": "system", "content": "You are a helpful assistant. Answer using ONLY the provided context. If not present, say you don't know."}
    user = {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    try:
        resp = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[system, user],
            max_tokens=500,
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Generation failed: {e}")
        return ""

# -------------------------
# UI: Multi-user + Upload + Select + Query + History
# -------------------------
# Session init
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "users_history" not in st.session_state:
    st.session_state["users_history"] = {}

# Login
def load_users() -> Dict[str, Dict[str,str]]:
    users = {}
    for i in range(1, 25):
        u = get_secret(f"USER_{i}_USERNAME")
        p = get_secret(f"USER_{i}_PASSWORD")
        r = get_secret(f"USER_{i}_ROLE")
        if u and p:
            users[u] = {"password": p, "role": r or "reader"}
    return users

def login_ui():
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
    login_ui()
    st.stop()

USER = st.session_state["user"]
ROLE = st.session_state.get("role", "reader")
st.title("🤖 SmartAssistant (RAG via Azure Search REST)")
st.caption(f"Signed in as: **{USER}** (role: {ROLE})")

# Clear conversation button (per-user)
if st.button("🧹 Clear Conversation"):
    st.session_state["users_history"][USER]["history"] = []
    st.success("Cleared conversation (UI only)")

# Upload (admin/editor only) - upload + index
if ROLE in ("admin", "editor"):
    uploaded = st.file_uploader("Upload PDF to index (PDF only)", type=["pdf"])
    if uploaded:
        st.info(f"Uploading & indexing {uploaded.name} ...")
        ok = upload_and_index_pdf(uploaded)
        if ok:
            st.success("Upload + indexing completed")
            st.session_state["users_history"][USER]["history"].append(f"Uploaded {uploaded.name}")
else:
    st.info("Upload disabled for readers")

st.divider()

# Document selection (one/multiple/all)
st.subheader("Select documents to search (leave empty = all)")
try:
    all_metadata = rest_get_unique_metadata(top=1000)
except Exception:
    all_metadata = []
selected_docs = st.multiselect("Choose documents (filename)", options=all_metadata, default=[])

# Query input
st.subheader("Ask a question about indexed documents")
query = st.text_input("Your question:")

if st.button("Run Query"):
    if not query.strip():
        st.error("Enter a question")
    else:
        # If user selected docs, pass list, else None (search all)
        selected = selected_docs if selected_docs else None
        with st.spinner("Retrieving relevant chunks..."):
            try:
                docs = retrieve_top_chunks(query, k=8, selected_docs=selected)
            except TypeError:
                # backward compat - some earlier function signature names
                docs = retrieve_top_chunks(query, k=8, selected_docs=selected)
            docs = dedupe_chunks(docs)
        if not docs:
            st.warning("No relevant documents found.")
        else:
            # build answer
            answer = generate_answer(query, docs)
            if answer:
                st.subheader("📝 Answer")
                st.write(answer)
                st.subheader("📄 Source Chunks")
                for i, d in enumerate(docs, start=1):
                    st.markdown(f"**Source #{i}:** {d.get('metadata','unknown')} (page {d.get('page')})")
                    st.write(d.get("content")[:1500])
                    st.markdown("---")
                st.session_state["users_history"][USER]["history"].append(f"Q: {query} -> A preview: {answer[:120]}")

# Show per-user history
st.subheader("Your session history (private)")
history_list = st.session_state["users_history"].get(USER, {}).get("history", [])
for h in history_list:
    st.write("- ", h)

# End of file
