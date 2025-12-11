import streamlit as st
import os
for proxy in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    if proxy in os.environ:
        del os.environ[proxy]

import openai
import tempfile

openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# NEW LangChain chain imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load ENV Vars
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
AZURE_BLOB_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME", "smartassistant-index")

if not all([
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_ADMIN_KEY, AZURE_SEARCH_INDEX_NAME,
    AZURE_BLOB_CONNECTION_STRING
]):
    st.error("🚨 Missing required environment variables.")
    st.stop()

# Azure Clients
blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
blob_container_client = blob_service_client.get_container_client(AZURE_BLOB_CONTAINER_NAME)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
)

# LLM
llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    api_version="2024-02-15-preview",
    temperature=0,
    max_tokens=500
)

# Embeddings
embeddings = AzureOpenAIEmbeddings(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    api_version="2024-02-15-preview"
)

# Vector Store
vectorstore = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_ADMIN_KEY,
    index_name=AZURE_SEARCH_INDEX_NAME,
    embedding_function=embeddings.embed_query
)

# === RAG PROMPT ===
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant that answers questions using ONLY the information in the provided context.

If the exact answer is NOT stated in the context, you may provide:
- An approximation or inferred answer, OR  
- A calculation based on available information  

…but you must clearly state that it is an estimation.

If NO estimation is possible, then respond:
"No information available in SOP documents."

Context:
{context}

Question:
{question}

Answer:
"""
)

st.title("🤖 SmartAssistantApp: SOP GenAI")
st.markdown("Search your SOPs with GenAI — Upload PDF/TXT/DOCX and ask questions.")

# -------------------------------------
# FILE UPLOAD
# -------------------------------------
st.header("📄 Upload New SOP Document")
uploaded_file = st.file_uploader("Upload SOP", type=["pdf", "txt", "docx"])

if uploaded_file:
    file_name = uploaded_file.name
    extension = file_name.split(".")[-1].lower()

    local_path = f"/tmp/{file_name}"
    with open(local_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    blob_container_client.upload_blob(file_name, uploaded_file, overwrite=True)
    st.success(f"Uploaded `{file_name}` to Azure Blob Storage.")

    # Select loader
    if extension == "pdf":
        loader = PyPDFLoader(local_path)
    elif extension == "txt":
        loader = TextLoader(local_path)
    else:
        loader = UnstructuredFileLoader(local_path)

    try:
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)

        # FIX: metadata must contain file_name (your index field)
        for doc in docs:
            doc.metadata = {"file_name": file_name}

        vectorstore.add_documents(docs)
        st.success(f"Indexed `{file_name}` ({len(docs)} chunks).")

    except Exception as e:
        st.error(f"Error processing document: {str(e)}")

# -------------------------------------
# LIST BLOB FILES
# -------------------------------------
st.header("📄 Available SOP Files")
blobs = [b.name for b in blob_container_client.list_blobs()]
st.write(blobs if blobs else "No files uploaded.")

# -------------------------------------
# RAG QUERY SECTION
# -------------------------------------
st.header("🔍 Query SOP Documents")

# Session state for clearing context
if "query_result" not in st.session_state:
    st.session_state.query_result = None
if "source_docs" not in st.session_state:
    st.session_state.source_docs = None

query_scope = st.selectbox("Query on:", ["All Documents"] + blobs)
question = st.text_input("Enter your question:")

col1, col2 = st.columns(2)
run_query = col1.button("Run Query")
reset_query = col2.button("Reset")

# Reset clears session state
if reset_query:
    st.session_state.query_result = None
    st.session_state.source_docs = None
    st.success("Query reset successfully.")
    st.stop()

# Execute ONLY when Run Query button is pressed
if run_query:

    if not question.strip():
        st.warning("Please enter a question before running query.")
        st.stop()

    # -------- RETRIEVER --------
    if query_scope != "All Documents":
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"filter": f"file_name eq '{query_scope}'"}
        )
        retriever.k = 5
    else:
        retriever = vectorstore.as_retriever(
            search_type="hybrid",
            search_kwargs={}
        )
        retriever.k = 5

    # ---------------------------------------------------------
    # 🔥 TIMEOUT-PROTECTED RETRIEVER BLOCK (NEW)
    # ---------------------------------------------------------
    with st.spinner("Searching SOPs..."):

        import threading

        result_holder = {"docs": None, "error": None}

        def run_retriever():
            try:
                result_holder["docs"] = retriever.get_relevant_documents(question)
            except Exception as e:
                result_holder["error"] = e

        t = threading.Thread(target=run_retriever)
        t.start()
        t.join(timeout=7)  # <-- 7-second timeout

        # If still running → timeout
        if t.is_alive():
            result_holder["error"] = TimeoutError("Retriever timed out")
            result_holder["docs"] = []

        # Any retriever error → safe fallback
        if result_holder["error"] is not None:
            st.session_state.query_result = "No information available in SOP documents."
            st.session_state.source_docs = []
            st.stop()

        docs = result_holder["docs"]

        # No docs returned → safe fallback
        if not docs:
            st.session_state.query_result = "No information available in SOP documents."
            st.session_state.source_docs = []
            st.stop()

    # ---------------------------------------------------------
    # END TIMEOUT BLOCK
    # ---------------------------------------------------------

    # -------- DEDUPLICATION --------
    unique_docs = []
    seen = set()
    for d in docs:
        snippet = d.page_content[:200]
        if snippet not in seen:
            seen.add(snippet)
            unique_docs.append(d)
    docs = unique_docs

    # Build full context
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # LLM call
    prompt = RAG_PROMPT.format(
        context=context_text,
        question=question
    )

    answer_obj = llm.invoke(prompt)
    answer = answer_obj.content

    st.session_state.query_result = answer
    st.session_state.source_docs = docs

# SHOW OUTPUT (only when query executed)
if st.session_state.query_result is not None:
    st.subheader("📝 Answer")
    st.write(st.session_state.query_result)

    if st.session_state.source_docs:
        st.subheader("📌 Source Chunks")
        for doc in st.session_state.source_docs:
            st.write(doc.page_content[:500])
