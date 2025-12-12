import streamlit as st
import os
import threading
import time

# Remove corporate proxies
for proxy in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(proxy, None)

# Standard imports for Azure + LangChain + embeddings
import openai
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# try numpy for faster cosine; fallback to pure python
try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

# -------------------------------
# ENVIRONMENT VARIABLES
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

# Validate environment
if not all([
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_ADMIN_KEY, AZURE_SEARCH_INDEX_NAME,
    AZURE_BLOB_CONNECTION_STRING
]):
    st.error("🚨 Missing required environment variables. Please verify your environment settings.")
    st.stop()

# -------------------------------
# CLIENT INIT
# -------------------------------
blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
blob_container_client = blob_service_client.get_container_client(AZURE_BLOB_CONTAINER_NAME)

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
    max_tokens=500
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
# RAG PROMPT
# -------------------------------
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant that answers questions using ONLY the information in the provided context.

If the exact answer is NOT stated in the context:
- Provide an estimation or inference only if it is clearly supported by the context.
- Otherwise respond exactly: "No information available in SOP documents."

Context:
{context}

Question:
{question}

Answer:
"""
)

# -------------------------------
# helper: cosine similarity
# -------------------------------
def cosine_similarity(a, b):
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

# -------------------------------
# UI start
# -------------------------------
st.title("🤖 SmartAssistantApp: SOP GenAI")
st.markdown("Upload SOP documents and query them using AI-powered search.")

# -------------------------------
# Upload & Index
# -------------------------------
st.header("📄 Upload New SOP Document")
uploaded_file = st.file_uploader("Upload SOP (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])

if uploaded_file:
    file_name = uploaded_file.name
    extension = file_name.split(".")[-1].lower()
    local_path = f"/tmp/{file_name}"
    with open(local_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        blob_container_client.upload_blob(file_name, uploaded_file, overwrite=True)
        st.success(f"Uploaded `{file_name}` to Azure Blob Storage.")
    except Exception as e:
        st.error(f"Blob upload error: {e}")

    # loader selection
    if extension == "pdf":
        loader = PyPDFLoader(local_path)
    elif extension == "txt":
        loader = TextLoader(local_path)
    else:
        loader = UnstructuredFileLoader(local_path)

    try:
        docs_raw = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(docs_raw)
        for d in docs:
            d.metadata = {"file_name": file_name}
        vectorstore.add_documents(docs)
        st.success(f"Indexed `{file_name}` with {len(docs)} chunks.")
    except Exception as e:
        st.error(f"Error indexing document: {e}")

# -------------------------------
# Show available files
# -------------------------------
st.header("📄 Available SOP Files")
try:
    blobs = [b.name for b in blob_container_client.list_blobs()]
except Exception:
    blobs = []
st.write(blobs if blobs else "No documents uploaded yet.")

# -------------------------------
# Query section
# -------------------------------
st.header("🔍 Query SOP Documents")

# initialize session keys if missing
if "query_result" not in st.session_state:
    st.session_state.query_result = None
if "source_docs" not in st.session_state:
    st.session_state.source_docs = None

# Query scope + text input
query_scope = st.selectbox("Search Scope:", ["All Documents"] + blobs)
question = st.text_input("Enter your question:", key="user_question")

col1, col2 = st.columns(2)
run_query = col1.button("Run Query")
reset_btn = col2.button("Reset")

# Reset behavior: clear session and rerun (safe)
if reset_btn:
    st.session_state.clear()
    # ensure there are no lingering widget keys — rerun to recreate widgets clean
    st.rerun()

# Run query logic
if run_query:
    if not question or not question.strip():
        st.warning("Please type a question.")
        st.stop()

    # choose retriever
    if query_scope == "All Documents":
        retriever = vectorstore.as_retriever(search_type="similarity")
    else:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"filter": f"file_name eq '{query_scope}'"}
        )
    retriever.k = 5

    # threaded retrieval with spinner/progress
    result_holder = {"docs": None, "error": None, "done": False}

    def retrieve():
        try:
            result_holder["docs"] = retriever.get_relevant_documents(question)
        except Exception as e:
            result_holder["error"] = e
        finally:
            result_holder["done"] = True

    with st.spinner("🔍 Searching SOPs..."):
        t = threading.Thread(target=retrieve)
        t.start()

        start_time = time.time()
        timeout_seconds = 8
        while time.time() - start_time < timeout_seconds:
            if result_holder["done"]:
                break
            time.sleep(0.1)

        if not result_holder["done"]:
            result_holder["error"] = TimeoutError("Retriever timed out")
            result_holder["docs"] = []
            result_holder["done"] = True

    # normalize
    docs_raw = result_holder.get("docs") or []

    # If retriever returned error or no docs -> out-of-scope fallback
    if result_holder.get("error") is not None or not docs_raw:
        st.session_state.query_result = "No information available in SOP documents."
        st.session_state.source_docs = []
        st.stop()

    # deduplicate preliminary
    unique_docs = []
    seen = set()
    for d in docs_raw:
        text = getattr(d, "page_content", None) or getattr(d, "content", None) or ""
        snip = text[:200]
        if snip not in seen:
            seen.add(snip)
            # ensure page_content is set
            if not getattr(d, "page_content", None) and getattr(d, "content", None):
                d.page_content = d.content
            unique_docs.append(d)

    if not unique_docs:
        st.session_state.query_result = "No information available in SOP documents."
        st.session_state.source_docs = []
        st.stop()

    # -------------------------------
    # OUT-OF-SCOPE DETECTION: manual similarity using embeddings.embed_query()
    # -------------------------------
    SIMILARITY_THRESHOLD = 0.55
    q_vec = None
    try:
        q_vec = embeddings.embed_query(question)
    except Exception as e:
        # Embedding failed — warn and allow LLM to attempt answer (conservative)
        st.warning("Warning: failed to embed question for OOS detection; proceeding without strict OOS check.")
        q_vec = None

    if q_vec is not None:
        max_sim = 0.0
        scored = []
        for d in unique_docs:
            text = d.page_content or ""
            try:
                doc_vec = embeddings.embed_query(text)
            except Exception:
                doc_vec = None
            sim = cosine_similarity(q_vec, doc_vec) if (doc_vec is not None) else 0.0
            scored.append((d, sim))
            if sim > max_sim:
                max_sim = sim

        # If highest similarity below threshold -> out-of-scope
        if max_sim < SIMILARITY_THRESHOLD:
            st.session_state.query_result = "No information available in SOP documents."
            st.session_state.source_docs = []
            st.stop()

        # else sort docs by similarity
        scored.sort(key=lambda x: x[1], reverse=True)
        docs = [s[0] for s in scored]
    else:
        # No q_vec → fallback to using the deduped unique_docs (not ideal, but safe)
        docs = unique_docs

    # Build context from top-K (docs list already limited by retriever.k)
    context_text = "\n\n".join([getattr(d, "page_content", "") for d in docs])

    # LLM call
    final_prompt = RAG_PROMPT.format(context=context_text, question=question)
    try:
        llm_answer = llm.invoke(final_prompt).content
    except Exception as e:
        st.error(f"LLM call error: {e}")
        llm_answer = "No information available in SOP documents."

    st.session_state.query_result = llm_answer
    st.session_state.source_docs = docs

# -------------------------------
# Show output
# -------------------------------
if st.session_state.get("query_result") is not None:
    st.subheader("📝 Answer")
    st.write(st.session_state.query_result)

    if st.session_state.get("source_docs"):
        st.subheader("📌 Source Chunks")
        for d in st.session_state.source_docs:
            st.write(d.page_content[:500])
