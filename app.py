import streamlit as st
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Research Paper Q&A Agent", page_icon="🔬", layout="centered")
st.title("🔬 Research Paper Q&A Agent")
st.write("Upload a research paper, ask questions, and get answers grounded strictly in the text.")

with st.sidebar:
    st.header("Configuration")
    st.markdown("### How it works:")
    st.caption("1. Chunk PDF into overlapping segments.")
    st.caption("2. Embed segments locally using HuggingFace.")
    st.caption("3. Index in a local FAISS Vector Store.")
    st.caption("4. Query uses strict system prompts to prevent hallucinations.")

if not os.environ.get("GROQ_API_KEY"):
    st.error("No API key found. Please add GROQ_API_KEY to your .env file.")
    st.stop()

@st.cache_resource(show_spinner="Processing and indexing document...")
def process_pdf(uploaded_file):
    temp_filename = "temp_uploaded_paper.pdf"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    loader = PyPDFLoader(temp_filename)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(splits, embeddings_model)
    
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type=["pdf"])

if uploaded_file is not None:
    retriever = process_pdf(uploaded_file)
    st.success("Document indexed successfully! You can now ask questions below.", icon="✅")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    system_prompt = (
        "You are an expert research assistant. Answer the user's question using ONLY the provided context below.\n\n"
        "CRITICAL RULE: If the context does not contain the answer, or if you are unsure, state clearly: "
        "'I cannot find the answer in the provided document.' Do not attempt to make up an answer, fill in blanks, "
        "or use outside knowledge.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    user_query = st.text_input("Ask something about the paper:", placeholder="e.g., What dataset did this paper use?")
    
    if user_query:
        with st.spinner("Analyzing document chunks..."):
            try:
                response_text = rag_chain.invoke(user_query)
                st.markdown("### 🤖 Answer:")
                st.write(response_text)
            except Exception as e:
                st.error(f"Error: {e}")
