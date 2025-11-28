# app.py
import os
import uuid
import streamlit as st
from pydantic import BaseModel
from dotenv import load_dotenv

from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorQuery

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings

from openai import AzureOpenAI


# ---------------------------------------------------------
# SAFE SECRET LOADING
# ---------------------------------------------------------
def get_secret(key: str):
    # Attempt Streamlit secrets only if available AND safe
    try:
        if hasattr(st, "secrets") and st.secrets and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    # Fallback to environment variables
    return os.environ.get(key)


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="SmartAssistant App", layout="wide")

load_dotenv()


# ---------------------------------------------------------
# AUTHENTICATION (simple username/password)
# ---------------------------------------------------------
def login():
    st.title("🔐 SmartAssistant Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if username == get_secret("ADMIN_USERNAME") and password == get_secret("ADMIN_PASSWORD"):
            st.session_state["authenticated"] = True
            st.session_state["user"] = username
            st.session_state["chat_history"] = []     # User-specific chat history
            st.success("Login successful!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")


if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    login()
    st.stop()


# ---------------------------------------------------------
# LOAD CONFIG
# ---------------------------------------------------------
AZURE_OPENAI_ENDPOINT = get_secret("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = get_secret("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = get_secret("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = get_secret("AZURE_OPENAI_API_VERSION")

AZURE_SEARCH_ENDPOINT = get_secret("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = get_secret("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = get_secret("AZURE_SEARCH_INDEX")

AZURE_STORAGE_CONNECTION = get_secret("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = get_secret("AZURE_STORAGE_CONTAINER")


# ---------------------------------------------------------
# OPENAI CLIENT
# ---------------------------------------------------------
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)


# ---------------------------------------------------------
# EMBEDDINGS
# ---------------------------------------------------------
embedder = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)


# ---------------------------------------------------------
# AZURE SEARCH CLIENT
# ---------------------------------------------------------
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AZURE_SEARCH_KEY
)


# ---------------------------------------------------------
# UPLOAD TO BLOB + INDEXING
# ---------------------------------------------------------
def upload_pdf_to_blob(pdf_file):
    try:
        blob_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION)
        container = blob_client.get_container_client(AZURE_STORAGE_CONTAINER)

        blob_name = f"{uuid.uuid4()}-{pdf_file.name}"
        container.upload_blob(blob_name, pdf_file.read(), overwrite=True)

        return blob_name
    except Exception as e:
        st.error(f"Blob upload failed: {e}")
        return None


# ---------------------------------------------------------
# VECTOR RETRIEVAL (using new SDK)
# ---------------------------------------------------------
def retrieve_top_chunks(query: str, k: int = 5):
    try:
        query_vector = embedder.embed_query(query)

        results = search_client.search(
            search_text=None,
            vector_queries=[
                VectorQuery(
                    vector=query_vector,
                    k_nearest_neighbors=k,
                    fields="contentVector"
                )
            ],
            select=["content", "source", "chunk_id"]
        )

        docs = []
        for result in results:
            docs.append({
                "content": result["content"],
                "source": result.get("source", "Unknown"),
                "chunk_id": result.get("chunk_id", "")
            })

        return docs

    except Exception as e:
        st.error(f"Retrieval failed: {e}")
        return []


# ---------------------------------------------------------
# LLM ANSWERING
# ---------------------------------------------------------
def answer_query(user_query, chunks):

    # Combine chunk text
    combined_context = "\n\n".join([c["content"] for c in chunks])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer only from the provided context. "
                "If not in context, say you don't know."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{combined_context}\n\nQuestion: {user_query}"
        }
    ]

    try:
        completion = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            max_tokens=400,
            temperature=0
        )

        return completion.choices[0].message.content

    except Exception as e:
        st.error(f"Generation failed: {e}")
        return None


# ---------------------------------------------------------
# REMOVE DUPLICATES
# ---------------------------------------------------------
def dedupe_chunks(chunks):
    seen = set()
    deduped = []

    for c in chunks:
        text = c["content"].strip()
        if text not in seen:
            deduped.append(c)
            seen.add(text)

    return deduped


# ---------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------
st.title("🧠 SmartAssistant – RAG over Azure Search & OpenAI")
st.caption(f"Logged in as: **{st.session_state['user']}**")

# Feature 1 – CLEAR BUTTON
if st.button("🧹 Clear Conversation"):
    st.session_state["chat_history"] = []
    st.success("Chat cleared!")
    st.experimental_rerun()

# Upload PDF
uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_pdf:
    blob_name = upload_pdf_to_blob(uploaded_pdf)
    if blob_name:
        st.success(f"Uploaded {uploaded_pdf.name} to blob: {blob_name}")

st.divider()

# Query Section
user_query = st.text_input("Ask a question about your uploaded documents:")

if st.button("Run Query"):
    if not user_query.strip():
        st.error("Enter a question.")
    else:
        with st.spinner("Retrieving and generating answer..."):

            results = retrieve_top_chunks(user_query, k=5)
            results = dedupe_chunks(results)

            if not results:
                st.error("No relevant chunks found.")
            else:
                answer = answer_query(user_query, results)

                if answer:
                    # Record chat per user
                    st.session_state["chat_history"].append({
                        "q": user_query,
                        "a": answer,
                        "chunks": results
                    })


# ---------------------------------------------------------
# SHOW CHAT HISTORY FOR THIS USER ONLY
# ---------------------------------------------------------
st.subheader("📝 Conversation History")

for item in st.session_state["chat_history"]:
    st.markdown(f"**You:** {item['q']}")
    st.markdown(f"**Assistant:** {item['a']}")
    with st.expander("📄 Source Chunks"):
        for c in item["chunks"]:
            st.write(f"**Source:** {c['source']}")
            st.write(c["content"])
            st.markdown("---")
