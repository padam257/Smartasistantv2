import streamlit as st
import os
import openai
import tempfile
openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🌐 Load environment variable
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
AZURE_BLOB_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME", "smartassistant-sops")

# 🔷 Validate config
if not all([
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_ADMIN_KEY, AZURE_SEARCH_INDEX_NAME,
    AZURE_BLOB_CONNECTION_STRING
]):
    st.error("🚨 One or more required environment variables are missing.")
    st.stop()

# 🔷 Azure clients
blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
blob_container_client = blob_service_client.get_container_client(AZURE_BLOB_CONTAINER_NAME)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
)

# 🔷 LangChain components
llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    temperature=0,
    max_tokens=500
)

embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
)

vectorstore = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_ADMIN_KEY,
    index_name=AZURE_SEARCH_INDEX_NAME,
    embedding_function=embeddings.embed_query,
)

retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# 🌟 UI
st.title("🤖 SmartAssistantApp: SOP GenAI")
st.markdown("Query your SOPs using GenAI. Upload PDFs, view existing, and query all or specific.")

st.header("📄 Upload New SOP PDF")
uploaded_file = st.file_uploader("Upload SOP", type=["pdf", "txt", "docx"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_ext = file_name.split(".")[-1].lower()
    local_path = f"/tmp/{file_name}"

    # 🔹 Save to local temp path
    with open(local_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 🔹 Upload to Azure Blob Storage
    blob_container_client.upload_blob(file_name, uploaded_file, overwrite=True)
    st.success(f"✅ Uploaded `{file_name}` to Blob")

    # 🔹 Select loader based on file extension
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

        # 🧹 Clean metadata
        for doc in docs:
            flat_meta = {}
            for k, v in doc.metadata.items():
                if k == "metadata" and isinstance(v, dict):
                    flat_meta.update(v)
                else:
                    flat_meta[k] = v

            allowed_keys = {"source", "page", "metadata_storage_name"}
            filtered_meta = {k: flat_meta.get(k, "") for k in allowed_keys}
            filtered_meta["source"] = str(filtered_meta.get("source", ""))
            try:
                filtered_meta["page"] = int(filtered_meta.get("page", 0))
            except:
                filtered_meta["page"] = 0
            doc.metadata = filtered_meta

        for doc in docs[:3]:
            st.text("✅ Example document to be pushed:")
            st.code(doc.page_content[:200])
            st.text("Debug metadata sample:")
            st.json(doc.metadata)

        vectorstore.add_documents(docs)
        st.success(f"✅ Successfully indexed `{file_name}` with {len(docs)} chunks.")

    except Exception as e:
        st.error(f"❌ Failed to load or process document: {str(e)}")

# 📄 Show files in Blob
st.header("📄 Available SOPs in Blob Storage")
blobs = list(blob_container_client.list_blobs())
doc_names = [blob.name for blob in blobs]
if not doc_names:
    st.info("No SOPs uploaded yet.")
else:
    st.write("Available SOP PDFs:")
    for doc in doc_names:
        st.markdown(f"- {doc}")

# 🔍 Query Section
st.header("🔍 Query SOPs documents")
query_scope = st.selectbox("Run query on:", ["All Documents"] + doc_names, index=0)
user_query = st.text_input("Your question:")

if user_query:
    if query_scope != "All Documents":
        retriever = vectorstore.as_retriever(
            search_kwargs={"filter": f"metadata_storage_name eq '{query_scope}'"}
        )
    else:
        retriever = vectorstore.as_retriever()

    with st.spinner("Fetching answer..."):
        result = qa_chain.run(user_query)

    st.markdown("### 📝 Answer:")
    st.write(result)

    st.markdown("### 📄 Source Chunks:")
    for doc in result.get("source_documents", []):
        st.write(doc.page_content[:500])
