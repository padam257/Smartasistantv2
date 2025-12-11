import streamlit as st
import os
import threading

# Remove corporate proxies
for proxy in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(proxy, None)

import openai
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate


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

if not all([
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_ADMIN_KEY, AZURE_SEARCH_INDEX_NAME,
    AZURE_BLOB_CONNECTION_STRING
]):
    st.error("🚨 Missing required environment variables. Please verify your .env or environment settings.")
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

If the exact answer is NOT stated in the context, you may:
- Provide an estimation based on available info, OR  
- Infer the answer

…but you MUST clearly state it is an estimation.

If no relevant information exists, reply:
"No information available in SOP documents."

Context:
{context}

Question:
{question}

Answer:
"""
)


# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("🤖 SmartAssistantApp: SOP GenAI")
st.markdown("Upload SOP documents and query them using AI-powered search.")


# -------------------------------
# DOCUMENT UPLOAD & INDEXING
# -------------------------------
st.header("📄 Upload New SOP Document")
uploaded_file = st.file_uploader("Upload SOP (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])

if uploaded_file:
    file_name = uploaded_file.name
    extension = file_name.split(".")[-1].lower()

    # Save temp
    local_path = f"/tmp/{file_name}"
    with open(local_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    blob_container_client.upload_blob(file_name, uploaded_file, overwrite=True)
    st.success(f"Uploaded `{file_name}` to Azure Blob Storage.")

    # Loader selection
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
        st.success(f"Indexed `{file_name}` with {len(docs)} text chunks.")

    except Exception as e:
        st.error(f"Error indexing document: {str(e)}")


# -------------------------------
# LIST FILES
# -------------------------------
st.header("📄 Available SOP Files")
blobs = [b.name for b in blob_container_client.list_blobs()]
st.write(blobs if blobs else "No documents uploaded yet.")


# -------------------------------
# QUERY SECTION
# -------------------------------
st.header("🔍 Query SOP Documents")

if "query_result" not in st.session_state:
    st.session_state.query_result = None
if "source_docs" not in st.session_state:
    st.session_state.source_docs = None

query_scope = st.selectbox("Search Scope:", ["All Documents"] + blobs)
#question = st.text_input("Enter your question:")
question = st.text_input("Enter your question:", key="user_question")

col1, col2 = st.columns(2)
run_query = col1.button("Run Query")
reset_query = col2.button("Reset")

if reset_query:
    st.session_state.query_result = None
    st.session_state.source_docs = None
    st.session_state.user_question = ""   # CLEAR TEXT BOX
    st.success("Query reset successfully.")


# -------------------------------
# RUN QUERY
# -------------------------------
# -------------------------------
# RUN QUERY (REPLACE EXISTING run_query BLOCK WITH THIS)
# -------------------------------
if run_query:

    if not question.strip():
        st.warning("Please type a question.")
        st.stop()

    # Build retriever
    if query_scope == "All Documents":
        retriever = vectorstore.as_retriever(
            search_type="hybrid",
            search_kwargs={}
        )
    else:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"filter": f"file_name eq '{query_scope}'"}
        )

    retriever.k = 5

    # Timeout wrapper for retriever
    result_holder = {"docs": None, "error": None}

    def retrieve():
        try:
            result_holder["docs"] = retriever.get_relevant_documents(question)
        except Exception as e:
            result_holder["error"] = e

    with st.spinner("Searching SOPs..."):
        t = threading.Thread(target=retrieve)
        t.start()
        t.join(timeout=7)

        # handle timeout: thread still alive
        if t.is_alive():
            result_holder["error"] = TimeoutError("Retriever timed out")
            result_holder["docs"] = []

    # --- SAFETY: ensure docs iterable
    if result_holder["docs"] is None:
        result_holder["docs"] = []

    # If retriever had an error -> show debug and fallback
    if result_holder["error"] is not None:
        st.error(f"Retriever error: {result_holder['error']}")
        st.session_state.query_result = "No information available in SOP documents."
        st.session_state.source_docs = []
        # Fall back to light debugging info below (will show index entries)
    elif not result_holder["docs"]:
        # No vector hits. Provide diagnostic info to help fix index/metadata problems.
        st.warning("No vector matches found for this query and scope. Showing diagnostics...")

        # 1) Show whether the index contains documents with the selected file name
        if query_scope != "All Documents":
            try:
                # run a light keyword search on the index to see if docs matching file_name exist
                hits = list(search_client.search(search_text="*", filter=f"file_name eq '{query_scope}'", top=5))
                st.write(f"Index hits for file_name == '{query_scope}': {len(hits)}")
                if hits:
                    st.write("Sample index document fields (first hit):")
                    # show fields (avoid large dumps)
                    sample = hits[0].get("content", None) or hits[0]
                    st.write(sample if isinstance(sample, str) else dict(hits[0]))
                else:
                    st.info("No index documents found matching that file_name. Possible mismatch between blob name and indexed metadata.")
            except Exception as e:
                st.write("Diagnostic search failed:", str(e))
        else:
            # All documents chosen but no vector hits: show how many indexed docs exist
            try:
                hits = list(search_client.search(search_text="*", top=5))
                st.write(f"Top {len(hits)} index entries (sample):")
                if hits:
                    st.write(dict(hits[0]))
            except Exception as e:
                st.write("Diagnostic search failed:", str(e))

        # Finally set "No information available" as the answer
        st.session_state.query_result = "No information available in SOP documents."
        st.session_state.source_docs = []
    else:
        # We have vector results — proceed
        # Ensure result_holder["docs"] is iterable list
        if result_holder["docs"] is None:
            result_holder["docs"] = []

        # Deduplicate
        unique_docs = []
        seen = set()
        for d in result_holder["docs"]:
            # guard: ensure doc has page_content
            content = getattr(d, "page_content", None) or getattr(d, "content", None) or ""
            snip = content[:200]
            if snip not in seen:
                seen.add(snip)
                # normalize doc to always have .page_content (some providers use .content)
                if not getattr(d, "page_content", None) and getattr(d, "content", None):
                    d.page_content = d.content
                unique_docs.append(d)

        docs = unique_docs

        # If after dedupe nothing remains
        if not docs:
            st.session_state.query_result = "No information available in SOP documents."
            st.session_state.source_docs = []
        else:
            # Build full context
            context_text = "\n\n".join([d.page_content for d in docs])

            # Debug: show how many chunks will be sent (optional, remove if noisy)
            st.write(f"Retrieved {len(docs)} chunk(s); sending to LLM...")

            # LLM invocation (use final_prompt)
            final_prompt = RAG_PROMPT.format(context=context_text, question=question)

            try:
                llm_answer = llm.invoke(final_prompt).content
            except Exception as e:
                st.error(f"LLM invocation error: {e}")
                llm_answer = "No information available in SOP documents."

            st.session_state.query_result = llm_answer
            st.session_state.source_docs = docs
