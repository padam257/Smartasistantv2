# app.py
import os
import uuid
import tempfile
import streamlit as st
from typing import List, Dict

from dotenv import load_dotenv

# Azure SDK
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorQuery

# OpenAI (Azure) client
from openai import AzureOpenAI

# LangChain wrappers for embeddings & text splitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF extraction (Option A)
from pypdf import PdfReader

# Document typing
from dataclasses import dataclass

# -------------------------
# Load .env locally (optional)
# -------------------------
load_dotenv()

# -------------------------
# Safe secret loader (Streamlit secrets or env)
# -------------------------
def get_secret(key: str):
    try:
        if hasattr(st, "secrets") and st.secrets and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key)


# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="SmartAssistant", layout="wide")

# Required env vars (used later)
AZURE_OPENAI_ENDPOINT = get_secret("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = get_secret("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = get_secret("AZURE_OPENAI_DEPLOYMENT_NAME")  # chat model deployment
AZURE_OPENAI_API_VERSION = get_secret("AZURE_OPENAI_API_VERSION") or get_secret("OPENAI_API_VERSION")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = get_secret("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")  # embedding deployment (text-embedding-3-small)

AZURE_SEARCH_ENDPOINT = get_secret("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = get_secret("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = get_secret("AZURE_SEARCH_INDEX")

AZURE_STORAGE_CONNECTION = get_secret("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = get_secret("AZURE_STORAGE_CONTAINER")

# minimal validation
missing = [k for k,v in {
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
}.items() if not v]

if missing:
    st.error("Missing required environment variables: " + ", ".join(missing))
    st.stop()


# -------------------------
# Multi-user loader from env (USER_1..USER_24)
# -------------------------
def load_users() -> Dict[str, Dict]:
    users = {}
    for i in range(1, 25):
        uname = get_secret(f"USER_{i}_USERNAME")
        pwd = get_secret(f"USER_{i}_PASSWORD")
        role = get_secret(f"USER_{i}_ROLE")
        if uname and pwd:
            users[uname] = {"password": pwd, "role": role or "reader"}
    return users


def login():
    st.title("🔐 SmartAssistant Login")
    users = load_users()
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        if username in users and password == users[username]["password"]:
            # initialize session containers
            st.session_state["authenticated"] = True
            st.session_state["user"] = username
            st.session_state["role"] = users[username]["role"]
            if "users_history" not in st.session_state:
                st.session_state["users_history"] = {}
            st.session_state["users_history"].setdefault(username, {"history": []})
            st.success(f"Welcome, {username} ({st.session_state['role']})")
            st.rerun()
        else:
            st.error("Invalid username or password")


# require login
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    login()
    st.stop()

# convenience local vars
USER = st.session_state["user"]
ROLE = st.session_state.get("role", "reader")
st.caption(f"Signed in as: **{USER}** (role: {ROLE})")

# -------------------------
# Initialize clients
# -------------------------
# OpenAI Azure client (for chat generation)
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)

# Embeddings object (LangChain wrapper) - ensure embedding deployment is an embedding model
embedder = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    api_version=AZURE_OPENAI_API_VERSION,
)

# Azure Search client (use AzureKeyCredential)
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY),
)

# Blob client
blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION)
blob_container = blob_service.get_container_client(AZURE_STORAGE_CONTAINER)


# -------------------------
# PDF extraction (pypdf) -> list of page texts
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
            txt = p.extract_text() or ""
        except Exception:
            txt = ""
        pages.append(txt)
    return pages


# -------------------------
# chunk pages into langchain Documents
# -------------------------
def chunk_pages_into_documents(pages: List[str], source_name: str, chunk_size: int = 1200, chunk_overlap: int = 50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []
    for page_num, page_text in enumerate(pages, start=1):
        base = {"page_content": page_text, "metadata": {"metadata": source_name, "page": page_num}}
        # create a simple Document-like dict for splitter - splitter accepts dicts with page_content+metadata
        # but langchain splitter usually expects Document objects; however many wrappers accept simple dicts.
        # Use splitter.split_documents with Document objects for reliability:
        try:
            from langchain_core.documents import Document as LC_Doc
            d = LC_Doc(page_content=page_text, metadata={"metadata": source_name, "page": page_num})
        except Exception:
            try:
                from langchain.schema import Document as LC_Doc
                d = LC_Doc(page_content=page_text, metadata={"metadata": source_name, "page": page_num})
            except Exception:
                # fallback to dict (less ideal)
                d = base
        split_docs = splitter.split_documents([d])
        # ensure metadata contains 'metadata' (filename) and 'page'
        for sd in split_docs:
            md = sd.metadata or {}
            md["metadata"] = md.get("metadata", source_name)
            md["page"] = int(md.get("page", 0) or 0)
            sd.metadata = md
            docs.append(sd)
    return docs


# -------------------------
# index documents to Azure Search (embedding + upload)
# -------------------------
def index_documents_to_azure_search(docs) -> (bool, str):
    if not docs:
        return True, "No docs to index"
    texts = [d.page_content for d in docs]
    try:
        vectors = embedder.embed_documents(texts)  # returns list[list[float]]
    except Exception as e:
        return False, f"Embedding failed: {e}"

    azure_docs = []
    for i, d in enumerate(docs):
        md = d.metadata or {}
        doc_id = str(uuid.uuid4())
        azure_docs.append({
            "id": doc_id,
            "content": d.page_content,
            "metadata": md.get("metadata", "unknown"),     # store filename in 'metadata' field (your index uses this)
            "page": int(md.get("page", 0) or 0),
            "content_vector": vectors[i],
        })
    try:
        result = search_client.upload_documents(documents=azure_docs)
        return True, "Uploaded"
    except Exception as e:
        return False, f"Search upload failed: {e}"


# -------------------------
# full upload + index pipeline
# -------------------------
def upload_and_index_pdf(uploaded_file) -> bool:
    # read bytes (streamlit file)
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
# retrieval (vector query using embedder)
# -------------------------
def retrieve_top_chunks(query: str, k: int = 5):
    try:
        qvec = embedder.embed_query(query)
    except Exception as e:
        st.error(f"Query embedding failed: {e}")
        return []

    try:
        results = search_client.search(
            search_text=None,
            vector_queries=[VectorQuery(vector=qvec, k_nearest_neighbors=k, fields="content_vector")],
            select=["id", "content", "metadata", "page"]
        )
    except Exception as e:
        st.error(f"Azure Search query failed: {e}")
        return []

    docs = []
    for r in results:
        docs.append({
            "id": r.get("id"),
            "content": r.get("content", ""),
            "source": r.get("metadata", "unknown"),
            "page": r.get("page", 0)
        })
    return docs


# -------------------------
# dedupe exact duplicates
# -------------------------
def dedupe_chunks(items: List[Dict]):
    seen = set()
    out = []
    for it in items:
        txt = (it.get("content") or "").strip()
        if not txt:
            continue
        if txt not in seen:
            seen.add(txt)
            out.append(it)
    return out


# -------------------------
# answer generator (use top 3 chunks)
# -------------------------
def generate_answer(question: str, docs: List[Dict]) -> str:
    context = "\n\n---\n\n".join([f"Source: {d['source']} (page {d.get('page')})\n{d['content']}" for d in docs[:3]])
    system = {"role": "system", "content": "You are a helpful assistant. Answer using ONLY the provided context. If not in context, say you don't know."}
    user_msg = {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    try:
        resp = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[system, user_msg],
            max_tokens=500,
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Generation failed: {e}")
        return ""


# -------------------------
# UI: main
# -------------------------
st.title("🤖 SmartAssistant (RAG)")

# Clear conversation
if st.button("🧹 Clear Conversation"):
    st.session_state["users_history"][USER]["history"] = []
    st.success("Cleared conversation (UI only)")
    st.rerun()

# Upload area (admin + editor only)
if ROLE in ("admin", "editor"):
    uploaded = st.file_uploader("Upload PDF to index", type=["pdf"])
    if uploaded:
        st.info(f"Uploading & indexing {uploaded.name} ...")
        ok = upload_and_index_pdf(uploaded)
        if ok:
            st.success("Upload + indexing complete")
            st.session_state["users_history"][USER]["history"].append(f"Uploaded {uploaded.name}")
else:
    st.info("Upload disabled for readers.")

st.divider()

# Query area
query = st.text_input("Ask a question about uploaded documents:")
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
            ans = generate_answer(query, results)
            st.subheader("📝 Answer")
            st.write(ans)
            st.subheader("📄 Source Chunks")
            for i, d in enumerate(results, start=1):
                st.markdown(f"**Source #{i}:** {d['source']} (page {d.get('page')})")
                st.write(d["content"][:1000])
                st.markdown("---")
            # save history
            st.session_state["users_history"][USER]["history"].append(f"Q: {query} -> A preview: {ans[:120]}")

# show user history
st.subheader("Your session history (private)")
history = st.session_state["users_history"].get(USER, {}).get("history", [])
for h in history:
    st.write("- ", h)
