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
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader, AzureBlobStorageFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# NEW LangChain chain imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# Load ENV
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


# === NEW LangChain RAG CHAIN ===
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant answering questions based on organizational SOP documents.

Use only the context below to answer the question.  
If answer not found, say "No information available in SOP documents."

Context:
{context}

Question:
{question}

Answer:
"""
)

document_chain = create_stuff_documents_chain(llm, RAG_PROMPT)


# UI Header
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

    # Upload to Blob
    blob_container_client.upload_blob(file_name, uploaded_file, overwrite=True)
    st.success(f"Uploaded `{file_name}` to Azure Blob Storage.")

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

            # Fix metadata
            for doc in docs:
                meta = doc.metadata or {}
                cleaned = {"source": meta.get("source", file_name)}
                doc.metadata = cleaned

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
# RAG QUERY
# -------------------------------------
st.header("🔍 Query SOP Documents")
query_scope = st.selectbox("Query on:", ["All Documents"] + blobs)
question = st.text_input("Enter your question:")

if question:
    # Retriever (filtered or full)
    if query_scope != "All Documents":
        retriever = vectorstore.as_retriever(
            search_kwargs={"filter": f"metadata_storage_name eq '{query_scope}'"}
        )
    else:
        retriever = vectorstore.as_retriever()

    rag_chain = create_retrieval_chain(retriever, document_chain)

    with st.spinner("Searching SOPs..."):
        result = rag_chain.invoke({"input": question})

    st.subheader("📝 Answer")
    st.write(result["answer"])

    st.subheader("📌 Source Chunks")
    for doc in result["context"]:
        st.write(doc.page_content[:500])



