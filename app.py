# full app.py - Option C (LangChain retriever + re-rank + delete docs)
import streamlit as st
import os
import threading
import time
from typing import List

# Remove common corporate proxies
for p in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(p, None)

# Azure + LangChain imports
import openai
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# try numpy for faster cosine
try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

# --------------------------
# CONFIG (env vars expected)
# --------------------------
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
AZURE_BLOB_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME", "smartassistant-index")

RETRIEVER_TOP_K = 10      # LangChain retriever top-K to fetch
FINAL_K = 5               # Number of chunks sent to LLM after re-ranking
SIMILARITY_THRESHOLD = 0.55  # out-of-scope threshold (your requested)

# quick env validation
if not all([
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_ADMIN_KEY, AZURE_SEARCH_INDEX_NAME,
    AZURE_BLOB_CONNECTION_STRING
]):
    st.error("🚨 Missing required environment variables. Please check your environment.")
    st.stop()

# --------------------------
# clients
# --------------------------
blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
blob_container_client = blob_service_client.get_container_client(AZURE_BLOB_CONTAINER_NAME)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY),
)

llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    api_version="2024-02-15-preview",
    temperature=0,
    max_tokens=500,
)

embeddings = AzureOpenAIEmbeddings(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    api_version="2024-02-15-preview",
)

vectorstore = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_ADMIN_KEY,
    index_name=AZURE_SEARCH_INDEX_NAME,
    embedding_function=embeddings.embed_query,
)

# --------------------------
# Prompt
# --------------------------
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant that answers questions using ONLY the provided context.

If context does NOT contain the answer, reply exactly:
"No information available in SOP documents."

Context:
{context}

Question:
{question}

Answer:
"""
)

# --------------------------
# helpers
# --------------------------
def cosine_similarity(a, b) -> float:
    if a is None or b is None:
        return 0.0
    if _HAS_NUMPY:
        a_np = np.array(a, dtype=float)
        b_np = np.array(b, dtype=float)
        denom = (np.linalg.norm(a_np) * np.linalg.norm(b_np)) + 1e-12
        return float(np.dot(a_np, b_np) / denom)
    else:
        try:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(y * y for y in b) ** 0.5
            denom = (norm_a * norm_b) + 1e-12
            return dot / denom
        except Exception:
            return 0.0

def embed_text_safe(text: str):
    """
    Try to use embed_documents if available (better for long text).
    Fallback to embed_query if not available.
    Returns vector (list of floats) or None on failure.
    """
    if not text:
        return None
    try:
        # prefer embed_documents when available
        if hasattr(embeddings, "embed_documents"):
            vecs = embeddings.embed_documents([text])
            if isinstance(vecs, list) and len(vecs) > 0:
                return vecs[0]
            # fallthrough to embed_query
        # fallback
        vec = embeddings.embed_query(text)
        return vec
    except Exception:
        try:
            return embeddings.embed_query(text)
        except Exception:
            return None

# --------------------------
# UI layout & state
# --------------------------
st.set_page_config(page_title="SmartAssistantApp", layout="centered")
st.title("🔎 Query SOP Documents")

if "query_result" not in st.session_state:
    st.session_state.query_result = None
if "source_docs" not in st.session_state:
    st.session_state.source_docs = None

# --------------------------
# Upload & Indexing UI
# --------------------------
st.header("📄 Upload New SOP Document")
uploaded_file = st.file_uploader("Upload SOP (pdf, txt, docx)", type=["pdf", "txt", "docx"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    local_path = f"/tmp/{file_name}"
    with open(local_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        blob_container_client.upload_blob(file_name, uploaded_file, overwrite=True)
        st.success(f"Uploaded `{file_name}` to blob storage.")
    except Exception as e:
        st.error(f"Blob upload failed: {e}")

    # choose loader
    ext = file_name.split(".")[-1].lower()
    if ext == "pdf":
        loader = PyPDFLoader(local_path)
    elif ext == "txt":
        loader = TextLoader(local_path)
    else:
        loader = UnstructuredFileLoader(local_path)

    try:
        docs_raw = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(docs_raw)
        # set file_name metadata so we can filter later
        for d in docs:
            d.metadata = {"file_name": file_name}
        vectorstore.add_documents(docs)
        st.success(f"Indexed `{file_name}` with {len(docs)} chunks.")
    except Exception as e:
        st.error(f"Indexing failed: {e}")

# --------------------------
# List files + delete UI
# --------------------------
st.header("📄 Available SOP Files")
try:
    blob_list = [b.name for b in blob_container_client.list_blobs()]
except Exception:
    blob_list = []

st.write(blob_list if blob_list else "No documents uploaded yet.")

st.markdown("### Remove documents from system (blob + index)")
to_delete = st.multiselect("Select documents to delete:", blob_list)

if st.button("Delete selected documents"):
    if not to_delete:
        st.warning("Select one or more files to delete.")
    else:
        deleted = []
        failed = []
        for fname in to_delete:
            # 1) delete blob
            try:
                blob_container_client.delete_blob(fname)
                deleted.append(fname)
            except Exception as e:
                failed.append((fname, f"blob error: {e}"))
                continue

            # 2) try to delete from search index:
            # We will attempt to find documents by file_name in index and delete them by id field.
            try:
                filter_clause = f"file_name eq '{fname}'"
                results = search_client.search(search_text="*", filter=filter_clause, top=1000)
                ids_to_delete = []
                for r in results:
                    # common key name is "id" but it might differ; try common candidates
                    candidate_id = None
                    for key in ("id", "Id", "document_id", "documentKey"):
                        if key in r:
                            candidate_id = r[key]
                            break
                    # If still None, try metadata fields
                    if candidate_id is None and "metadata" in r:
                        # If metadata is stored as simple string equal to filename
                        try:
                            if r["metadata"] == fname:
                                # try to get id from r if exists
                                if "id" in r:
                                    candidate_id = r["id"]
                        except Exception:
                            pass
                    if candidate_id:
                        ids_to_delete.append(candidate_id)

                # Delete found ids (if any)
                if ids_to_delete:
                    # Try delete_documents by key values
                    try:
                        search_client.delete_documents(key_values=ids_to_delete)
                    except Exception:
                        # fallback: try delete_documents with documents containing id field
                        try:
                            docs_for_delete = [{"id": v} for v in ids_to_delete]
                            search_client.delete_documents(documents=docs_for_delete)
                        except Exception as e2:
                            # can't delete; log and continue
                            failed.append((fname, f"index-delete-failed: {e2}"))
                else:
                    # nothing found to delete in index for this file
                    # not necessarily an error - it might be that index uses different metadata key
                    pass
            except Exception as e:
                failed.append((fname, f"index search error: {e}"))

        # show results
        if deleted:
            st.success(f"Deleted blobs: {deleted}")
        if failed:
            st.error(f"Failures: {failed}")

        # refresh blob_list (next rerun)
        st.experimental_rerun()

# --------------------------
# Query UI
# --------------------------
st.header("🔍 Query SOP Documents")
query_scope = st.selectbox("Search Scope:", ["All Documents"] + blob_list)
question = st.text_input("Enter your question:", key="user_question")

col1, col2 = st.columns(2)
run_query = col1.button("Run Query")
reset_btn = col2.button("Reset")

# Reset behavior (clear state and rerun)
if reset_btn:
    st.session_state.clear()
    st.rerun()

# --------------------------
# RUN query (Option C)
# --------------------------
if run_query:
    if not question or not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Use LangChain retriever to fetch top RETRIEVER_TOP_K
    if query_scope == "All Documents":
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={})
    else:
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"filter": f"file_name eq '{query_scope}'"}
        )
    retriever.k = RETRIEVER_TOP_K

    result_holder = {"docs": None, "error": None, "done": False}

    def do_retrieve():
        try:
            result_holder["docs"] = retriever.get_relevant_documents(question)
        except Exception as e:
            result_holder["error"] = e
        finally:
            result_holder["done"] = True

    with st.spinner("🔍 Searching SOPs..."):
        t = threading.Thread(target=do_retrieve)
        t.start()
        start_t = time.time()
        while time.time() - start_t < 10:
            if result_holder["done"]:
                break
            time.sleep(0.1)
        if not result_holder["done"]:
            result_holder["error"] = TimeoutError("Retriever timed out")
            result_holder["docs"] = []
            result_holder["done"] = True

    docs_raw = result_holder.get("docs") or []
    if result_holder.get("error") is not None or not docs_raw:
        st.session_state.query_result = "No information available in SOP documents."
        st.session_state.source_docs = []
        st.stop()

    # dedupe quick
    unique_docs = []
    seen = set()
    for d in docs_raw:
        text = getattr(d, "page_content", None) or getattr(d, "content", None) or ""
        snip = text[:250]
        if snip not in seen:
            seen.add(snip)
            if not getattr(d, "page_content", None) and getattr(d, "content", None):
                d.page_content = d.content
            unique_docs.append(d)

    if not unique_docs:
        st.session_state.query_result = "No information available in SOP documents."
        st.session_state.source_docs = []
        st.stop()

    # --------------------------
    # Re-rank retrieved docs using embeddings (document embeddings when available)
    # --------------------------
    # compute question embedding
    try:
        q_vec = embeddings.embed_query(question)
    except Exception:
        # fallback to None and allow LLM (less safe)
        q_vec = None

    scored = []
    max_sim = 0.0
    # compute doc embeddings individually using embed_documents if available
    for d in unique_docs:
        text = d.page_content or ""
        doc_vec = embed_text_safe(text)  # this will try embed_documents else embed_query
        sim = cosine_similarity(q_vec, doc_vec) if q_vec is not None and doc_vec is not None else 0.0
        scored.append((d, sim))
        if sim > max_sim:
            max_sim = sim

    # If highest sim below threshold => out-of-scope
    if max_sim < SIMILARITY_THRESHOLD:
        st.session_state.query_result = "No information available in SOP documents."
        st.session_state.source_docs = []
        st.stop()

    # re-order and pick top FINAL_K
    scored.sort(key=lambda x: x[1], reverse=True)
    top_docs = [tup[0] for tup in scored[:FINAL_K]]

    # Build context (concatenate top docs)
    context_text = "\n\n".join([getattr(d, "page_content", "") for d in top_docs])

    # Call LLM
    final_prompt = RAG_PROMPT.format(context=context_text, question=question)
    try:
        llm_response = llm.invoke(final_prompt)
        answer_text = getattr(llm_response, "content", None) or str(llm_response)
    except Exception as e:
        st.error(f"LLM call failed: {e}")
        answer_text = "No information available in SOP documents."

    st.session_state.query_result = answer_text
    st.session_state.source_docs = top_docs

# --------------------------
# SHOW OUTPUT
# --------------------------
if st.session_state.get("query_result") is not None:
    st.subheader("📝 Answer")
    st.write(st.session_state.query_result)

    if st.session_state.get("source_docs"):
        st.subheader("📌 Source Chunks (top results)")
        for idx, d in enumerate(st.session_state.source_docs, start=1):
            snippet = getattr(d, "page_content", "")[:600]
            st.write(f"{idx}. {snippet}")
