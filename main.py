import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from transformers import pipeline


# Streamlit UI
st.set_page_config(page_title="📘 Explainer", page_icon="📘", layout="centered")
st.title("📘 Doc Explainer")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    # Save uploaded file
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load & split document
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Embeddings and Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)
    
    # HuggingFace QnA model
    model_name = "google/flan-t5-base" 
    qa_pipeline = pipeline("text2text-generation", model=model_name, max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=qa_pipeline)


    # Custom prompt for cleaner answers(thanks gpt)

    template = """
    You are a helpful assistant. Use the following context to answer the question.
    If the answer is not in the document, say "I could not find that in the document."

    Context:
    {context}

    Question: {question}

    Answer:
    """
    qa_prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": qa_prompt}
    )


    # for query 
    query = st.text_input("Ask question:")

    if query:
        with st.spinner("Thinking..."):
            response = qa_chain.run(query)
        st.markdown("**Result:**")
        st.write(response)
