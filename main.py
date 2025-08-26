import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Document Explainer", page_icon="📚")
st.title("📚 Document Explainer")

uploaded_file = st.file_uploader("Upload a TXT or PDF file", type=["txt", "pdf"])

# --------------------------
# Process uploaded file
# --------------------------
documents = []

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    temp_path = os.path.join("temp_" + uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load documents based on file type
    if uploaded_file.name.endswith(".txt"):
        loader = TextLoader(temp_path, encoding="utf-8")
    elif uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(temp_path)
    else:
        st.error("Unsupported file type")
        st.stop()

    documents = loader.load()
    st.success(f"Loaded {len(documents)} document chunks.")

if not documents:
    st.info("Please upload a TXT or PDF file to continue.")
    st.stop()

# --------------------------
# Split documents
# --------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# --------------------------
# Embeddings & Vector Store
# --------------------------
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)

# --------------------------
# QA with LLM
# --------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

query = st.text_input("Ask a question about your document:")

if query:
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])
    prompt = f"Based on the document, answer the following:\n\nContext:\n{context}\n\nQuestion: {query}"

    response = llm.invoke(prompt)
    st.markdown("### Answer:")
    st.write(response.content)
