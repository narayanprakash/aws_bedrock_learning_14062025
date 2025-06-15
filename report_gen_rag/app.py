import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Bedrock
from langchain.chains import RetrievalQA
import os

# --- Initialize Bedrock ---
def get_bedrock_embeddings():
    return BedrockEmbeddings(
        region_name="us-east-1",
        model_id="amazon.titan-embed-text-v2:0"
    )

def get_bedrock_llm():
    return Bedrock(
        region_name="us-east-1",
        model_id="amazon.titan-text-premier-v1:0",
        model_kwargs={"maxTokenCount": 512,"temperature":0.1}
    )

# --- Streamlit UI ---
st.title("PDF Chat with Amazon Titan")
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    # Process PDF
    with st.spinner("Processing PDF..."):
        reader = PdfReader(uploaded_file)
        text = " ".join([page.extract_text() for page in reader.pages])
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        
        # Create FAISS index (with safety warning)
        embeddings = get_bedrock_embeddings()
        vector_db = FAISS.from_texts(chunks, embeddings)
        vector_db.save_local("faiss_index")

    # Question Answering
    question = st.text_input("Ask a question:")
    if question:
        with st.spinner("Thinking..."):
            # Load FAISS with explicit deserialization permission
            vector_db = FAISS.load_local(
                "faiss_index",
                get_bedrock_embeddings(),
                allow_dangerous_deserialization=True  # Critical fix
            )
            qa = RetrievalQA.from_chain_type(
                llm=get_bedrock_llm(),
                chain_type="stuff",
                retriever=vector_db.as_retriever()
            )
            answer = qa.run(question)
        st.write("**Answer:**", answer)