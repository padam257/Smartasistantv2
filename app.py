import streamlit as st
import os
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
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
AZURE_BLOB_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME", "sops")

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
st.title("🤖 SmartAssistantv2: SOP GenAI")
st.markdown("Query your SOPs using GenAI. Upload PDFs, view existing, and query all or specific.")

st.header("📄 Upload New SOP PDF")
uploaded_file = st.file_uploader("Upload SOP", type=["pdf"])
if uploaded_file:
    blob_container_client.upload_blob(uploaded_file.name, uploaded_file, overwrite=True)
    st.success(f"✅ Uploaded `{uploaded_file.name}`")

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
        retriever.filter = f"metadata_storage_name eq '{query_scope}'"
    else:
        retriever.filter = None

    with st.spinner("Fetching answer..."):
        result = qa_chain(user_query)

    st.markdown("### 📝 Answer:")
    st.write(result['result'])

    st.markdown("### 📄 Source Chunks:")
    for doc in result['source_documents']:
        st.write(doc.page_content[:500])  # Show first 500 chars of each
