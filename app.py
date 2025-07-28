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
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores.azuresearch import AzureSearch
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from pathlib import Path
from langchain.schema import Document
#from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.document_loaders import AzureBlobStorageFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.embeddings.openai import OpenAIEmbeddings
#from langchain_community.retrievers.azure_cognitive_search import AzureCognitiveSearchRetriever
#from langchain.retrievers.azure_cognitive_search import AzureCognitiveSearchRetriever

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
    #openai_api_key=AZURE_OPENAI_API_KEY,
    api_key=AZURE_OPENAI_API_KEY,
    #openai_api_base=AZURE_OPENAI_ENDPOINT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    #deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    temperature=0,
    max_tokens=500
)

#retriever = AzureCognitiveSearchRetriever(
#    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
#    azure_search_key=AZURE_SEARCH_ADMIN_KEY,
#    index_name=AZURE_SEARCH_INDEX_NAME
#)

embeddings = AzureOpenAIEmbeddings(
    # openai_api_key=AZURE_OPENAI_API_KEY,
    # openai_api_base=AZURE_OPENAI_ENDPOINT
    # openai_api_base=f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME}",
    #openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    #openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    #deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002"),
    #openai_api_type="azure"
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
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(local_path)
    elif file_ext == "txt":
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(local_path)
    elif file_ext == "docx":
        from langchain_community.document_loaders import UnstructuredFileLoader
        loader = UnstructuredFileLoader(local_path)
    else:
        st.error(f"Unsupported file type: {file_ext}")
        st.stop()

    # 🔹 Load, chunk and embed
    try:
        documents = loader.load()
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)

    # ✅ Remove unsupported metadata before uploading
     #   for doc in docs:
     #       doc.metadata = {
     #           k: v for k, v in doc.metadata.items()
     #           if k in ["metadata_storage_name"]  # Adjust based on your schema
     #       }
        
     # Clean and re-wrap
     #   cleaned_docs = []
     #  for doc in docs:
     #       cleaned_doc = Document(
     #           page_content=doc.page_content,
     #           metadata={
     #               "metadata_storage_name": doc.metadata.get("metadata_storage_name", "")
     #           }
     #       )
     #       cleaned_docs.append(cleaned_doc)   

        for i, doc in enumerate(docs):
    # Add enriched metadata for index
            doc.metadata["metadata_storage_name"] = uploaded_file.name
            doc.metadata["source"] = uploaded_file.name
            doc.metadata["page"] = str(i + 1)  # or doc.metadata.get("page", "1")
            
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
    # Optional: Filter by document if selected
    if query_scope != "All Documents":
    #    retriever.filter = f"metadata_storage_name eq '{query_scope}'"
         retriever = vectorstore.as_retriever(
             search_kwargs={
                 "filter": f"metadata_storage_name eq '{query_scope}'"
             }   
        )
    else:
    #    retriever.filter = None
        retriever = vectorstore.as_retriever()

    with st.spinner("Fetching answer..."):
        result = qa_chain(user_query)

    st.markdown("### 📝 Answer:")
    st.write(result['result'])

    st.markdown("### 📄 Source Chunks:")
    for doc in result['source_documents']:
        st.write(doc.page_content[:500])  # Show first 500 chars of each
