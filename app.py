import streamlit as st
import os
import openai
import tempfile
from pathlib import Path
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader

# 🔐 Environment Variables
openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
AZURE_BLOB_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME", "smartassistant-sops")

# 🔍 Validate Required Config
if not all([
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_ADMIN_KEY, AZURE_SEARCH_INDEX_NAME,
    AZURE_BLOB_CONNECTION_STRING
]):
    st.error("🚨 One or more required environment variables are missing.")
    st.stop()

# 🔷 Azure Clients
blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
blob_container_client = blob_service_client.get_container_client(AZURE_BLOB_CONTAINER_NAME)
search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY))

# 🔷 LangChain Setup
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

vectorstore = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_ADMIN_KEY,
    index_name=AZURE_SEARCH_INDEX_NAME,
    embedding_function=embeddings.embed_query
)

retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True)

# 📋 Streamlit UI
st.title("🤖 SmartAssistantApp: SOP GenAI")
st.markdown("Query your SOPs using GenAI. Upload PDFs, view existing, and query all or specific.")

# 📤 Upload
st.header("📄 Upload New SOP PDF")
uploaded_file = st.file_uploader("Upload SOP", type=["pdf", "txt", "docx"])

if uploaded_file:
    file_name = uploaded_file.name
    file_ext = file_name.split(".")[-1].lower()
    local_path = f"/tmp/{file_name}"

    with open(local_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    blob_container_client.upload_blob(file_name, uploaded_file, overwrite=True)
    st.success(f"✅ Uploaded `{file_name}` to Blob")

    if file_ext == "pdf":
        loader = PyPDFLoader(local_path)
    elif file_ext == "txt":
        loader = TextLoader(local_path)
    elif file_ext == "docx":
        loader = UnstructuredFileLoader(local_path)
    else:
        st.error(f"Unsupported file type: {file_ext}")
        st.stop()

    try:
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)

        from langchain.schema import Document

        flattened_docs = []
        for doc in docs:
            meta = doc.metadata or {}

    # Flatten nested metadata
            flat_meta = {}
            for k, v in meta.items():
                if k == "metadata" and isinstance(v, dict):
                    flat_meta.update(v)
                else:
                    flat_meta[k] = v

    # Only allowed fields
            allowed_keys = {"source", "page", "metadata_storage_name"}
            flat_meta = {k: v for k, v in flat_meta.items() if k in allowed_keys}

    # Type checks
            flat_meta["source"] = str(flat_meta.get("source", ""))
            try:
                flat_meta["page"] = int(flat_meta.get("page", 0))
            except Exception:
                flat_meta["page"] = 0

            flat_meta["metadata_storage_name"] = os.path.basename(local_path)

            # Create sanitized Document (no nested metadata key!)
            clean_doc = Document(
                page_content=doc.page_content,
                metadata=flat_meta
            )

            st.write("✅ Example document to be pushed:")
            st.write(clean_doc.page_content[:300])
            st.write("Metadata:", clean_doc.metadata)

            flattened_docs.append(clean_docs)


        vectorstore.add_documents(docs)
        st.success(f"✅ Successfully indexed `{file_name}` with {len(cleaned_docs)} chunks.")

    except Exception as e:
        st.error(f"❌ Failed to load or process document: {str(e)}")

# 📂 List Files
st.header("📄 Available SOPs in Blob Storage")
blobs = list(blob_container_client.list_blobs())
doc_names = [blob.name for blob in blobs]
if not doc_names:
    st.info("No SOPs uploaded yet.")
else:
    for doc in doc_names:
        st.markdown(f"- {doc}")

# 🔍 Query Section
st.header("🔍 Query SOPs documents")
query_scope = st.selectbox("Run query on:", ["All Documents"] + doc_names, index=0)
user_query = st.text_input("Your question:")

if user_query:
    if query_scope != "All Documents":
        retriever = vectorstore.as_retriever(search_kwargs={"filter": f"metadata_storage_name eq '{query_scope}'"})
    else:
        retriever = vectorstore.as_retriever()

    with st.spinner("Fetching answer..."):
        result = qa_chain(user_query)

    st.markdown("### 📝 Answer:")
    st.write(result['result'])

    st.markdown("### 📄 Source Chunks:")
    for doc in result['source_documents']:
        st.write(doc.page_content[:500])
