# app.py
import streamlit as st
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from azure.core.credentials import AzureKeyCredential


# -------------------------------------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------------------------------------

AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_MODEL = st.secrets["AZURE_OPENAI_MODEL"]
AZURE_OPENAI_EMBEDDING_MODEL = st.secrets["AZURE_OPENAI_EMBEDDING_MODEL"]

AZURE_SEARCH_ENDPOINT = st.secrets["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_KEY = st.secrets["AZURE_SEARCH_KEY"]
AZURE_SEARCH_INDEX = st.secrets["AZURE_SEARCH_INDEX"]


# -------------------------------------------------------------------------------------
# 2. INITIALIZE LLM AND EMBEDDINGS
# -------------------------------------------------------------------------------------

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    model=AZURE_OPENAI_MODEL,
    temperature=0.2,
    max_tokens=1000,
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    model=AZURE_OPENAI_EMBEDDING_MODEL,
)


# -------------------------------------------------------------------------------------
# 3. REMOVE DUPLICATE CHUNKS
# -------------------------------------------------------------------------------------

def dedupe_docs(docs):
    seen = set()
    unique = []
    for d in docs:
        txt = d.page_content.strip()
        if txt not in seen:
            seen.add(txt)
            unique.append(d)
    return unique


# -------------------------------------------------------------------------------------
# 4. DOCUMENT LOADING + CHUNKING
# -------------------------------------------------------------------------------------

def load_and_chunk(file):
    filename = file.name.lower()

    if filename.endswith(".pdf"):
        loader = PyPDFLoader(file)
    elif filename.endswith(".docx"):
        loader = Docx2txtLoader(file)
    else:
        raise Exception("Only PDF and DOCX files are supported.")

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=50,
    )

    chunks = text_splitter.split_documents(docs)
    return chunks


# -------------------------------------------------------------------------------------
# 5. VECTOR STORE CONNECTION
# -------------------------------------------------------------------------------------

def get_vector_store():
    vector_store = AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_KEY,
        index_name=AZURE_SEARCH_INDEX,
        embedding_function=embeddings.embed_query,
    )
    return vector_store


# -------------------------------------------------------------------------------------
# 6. STORE DOCUMENTS INTO VECTOR SEARCH
# -------------------------------------------------------------------------------------

def upload_to_vector_search(chunks):
    vs = get_vector_store()
    vs.add_documents(documents=chunks)
    return True


# -------------------------------------------------------------------------------------
# 7. CUSTOM RAG PROMPT
# -------------------------------------------------------------------------------------

template = """
You are an AI assistant answering questions strictly based on the provided context.

If the answer is not present in the context, reply:
"Information not available in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)


# -------------------------------------------------------------------------------------
# 8. RAG - RETRIEVER + LLM
# -------------------------------------------------------------------------------------

def get_answer(user_query):
    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    docs = retriever.invoke(user_query)

    docs = dedupe_docs(docs)
    docs = docs[:3]  # keep top 3 after dedupe

    context = "\n\n---\n\n".join([d.page_content for d in docs])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    response = chain.invoke({"query": user_query})

    return response, docs


# -------------------------------------------------------------------------------------
# 9. STREAMLIT UI
# -------------------------------------------------------------------------------------

st.title("📘 Smart Assistant – RAG powered by Azure OpenAI + Azure AI Search")
st.write("Upload PDF/DOCX → Vector Embed → Ask Questions")


# ====== FILE UPLOAD SECTION ======
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx"])

if uploaded_file:
    st.info("Processing document ...")
    chunks = load_and_chunk(uploaded_file)
    upload_to_vector_search(chunks)
    st.success("Document embedded into Azure AI Search ✔")


# ====== QUESTION SECTION ======
question = st.text_input("Ask a question based on uploaded documents")

if question:
    response, docs = get_answer(question)

    st.subheader("🔍 Answer")
    st.write(response["result"])

    st.subheader("📄 Sources Used")
    for i, d in enumerate(docs, 1):
        st.markdown(f"**Source #{i}:**")
        st.write(d.page_content)
        st.markdown("---")
