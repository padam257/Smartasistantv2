import streamlit as st
import os
import tempfile
from pathlib import Path

import openai
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch

from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
)

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ==========================================================
# 🔐 ENVIRONMENT VARIABLES
# ==========================================================
openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Load environment variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
AZURE_BLOB_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME", "smartassistant-sops")


# ==========================================================
# 🚨 VALIDATION
# ==========================================================
required_env = [
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_ADMIN_KEY, AZURE_SEARCH_INDEX_NAME,
    AZURE_BLOB_CONNECTION_STRING
]

if not all(required_env):
    st.error("🚨 One or more required environment variables are missing.")
    st.stop()


# ==========================================================
# 🔷 AZURE CLIENTS
# ==========================================================
blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
blob_container_client = blob_service_client.get_container_client(AZURE_BLOB_CONTAINER_NAME)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
)


# ==========================================================
# 🔷 LLM + EMBEDDINGS
# ==========================================================
llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    temperature=0,
    max_tokens=500
)

embeddings = AzureOpenAIEmbeddings(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
)


# ==========================================================
# 🔷 VECTOR STORE (Azure Search)
# ==========================================================
vectorstore = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_ADMIN_KEY,
    index_name=AZURE_SEARCH_INDEX_NAME,
    embedding_function=embeddings.embed_query
)

retriever = vectorstore.as_retriever()


# ==========================================================
# 🔷 RETRIEVAL QA CHAIN (NEW LANGCHAIN v0.1+)
# ==========================================================
prompt = PromptTemplate.from_template("""
You are an AI assistant. Use ONLY the following context to answer.

<context>
{context}
</context>

Question: {input}
Answer:
""")

document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, document_chain)


# ==========================================================
# 📋 STREAMLIT UI
# ==========================================================
st.title("🤖 SmartAssistantApp: SOP GenAI")
st.markdown("Query your SOPs using GenAI. Upload PDFs, view existing, and query documents.")


# ==========================================================
# 📤 UPLOAD NEW SOP
# ==========================================================
st.header("📄 Upload New SOP PDF/Text/Docx")

uploaded_file = st.file_uploader("Upload SOP", type=["pdf", "txt", "docx"])

if uploaded_file:
    file_name = uploaded_file.name
    file_ext = file_name.split(".")[-1].lower()
    local_path = f"/tmp/{file_name}"

    with open(local_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    blob_container_client.upload_blob(file_name, uploaded_file, overwrite=True)
    st.success(f"Uploaded `{file_name}` to blob storage.")

    # DOC LOADER
    if file_ext == "pdf":
        loader = PyPDFLoader(local_path)
    elif file_ext == "txt":
        loader = TextLoader(local_path)
    elif file_ext == "docx":
        loader = UnstructuredFileLoader(local_path)
    else:
        st.error("Unsupported file type")
        st.stop()

    try:
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)

        flattened_docs = []

        for doc in docs:
            meta = doc.metadata or {}
            flat_meta = {}

            for k, v in meta.items():
                if k == "metadata" and isinstance(v, dict):
                    flat_meta.update(v)
                else:
                    flat_meta[k] = v

            allowed_keys = {"source", "page", "metadata_storage_name"}
            flat_meta = {k: v for k, v in flat_meta.items() if k in allowed_keys}

            flat_meta["source"] = str(flat_meta.get("source", ""))
            try:
                flat_meta["page"] = int(flat_meta.get("page", 0))
            except:
                flat_meta["page"] = 0

            flat_meta["metadata_storage_name"] = file_name

            if "metadata" in flat_meta:
                del flat_meta["metadata"]

            clean_doc = Document(
                page_content=doc.page_content,
                metadata=flat_meta
            )

            flattened_docs.append(clean_doc)

        vectorstore.add_documents(flattened_docs)
        st.success(f"Indexed `{file_name}` with {len(flattened_docs)} chunks.")

    except Exception as e:
        st.error(f"❌ Failed: {str(e)}")


# ==========================================================
# 📂 LIST EXISTING DOCS
# ==========================================================
st.header("📄 Available SOPs")

blobs = list(blob_container_client.list_blobs())
doc_names = [blob.name for blob in blobs]

if not doc_names:
    st.info("No SOPs uploaded yet.")
else:
    st.write("Documents:")
    for doc in doc_names:
        st.markdown(f"- {doc}")


# ==========================================================
# 🔍 QUERY SECTION
# ==========================================================
st.header("🔍 Query SOP documents")

query_scope = st.selectbox("Run query on:", ["All Documents"] + doc_names, index=0)
user_query = st.text_input("Your question:")

if user_query:
    if query_scope != "All Documents":
        retriever = vectorstore.as_retriever(
            search_kwargs={"filter": f"metadata_storage_name eq '{query_scope}'"}
        )
    else:
        retriever = vectorstore.as_retriever()

    chain = create_retrieval_chain(retriever, document_chain)

    with st.spinner("Fetching answer..."):
        response = chain.invoke({"input": user_query})

    st.subheader("📝 Answer")
    st.write(response["answer"])

    st.subheader("📄 Source Chunks")
    for doc in response["context"]:
        st.write(doc.page_content[:500])




