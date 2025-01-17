import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import faiss
import tempfile  # For temporary file handling
from langchain.docstore.in_memory import InMemoryDocstore

st.title("📄🤖 PDF-based Chatbot For Conference Call Transcript")
st.write("This chatbot responds based on the content of uploaded PDFs using embeddings and vector storage.")

# Load environment variables
load_dotenv()

# Initialize model
model = ChatOllama(model='llama3.2:1b', base_url="http://localhost:11434")

# Define function to load and process PDFs
def load_and_chunk_pdfs(uploaded_files):
    all_chunks = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyMuPDFLoader(tmp_file_path)  # Load using the temporary file path
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        
        # Clean up the temporary file after loading
        os.remove(tmp_file_path)
        
        all_chunks.extend(chunks)  # Add chunks from this PDF to the list
    
    return all_chunks

# Upload multiple PDF files
uploaded_pdfs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
if uploaded_pdfs:
    st.session_state["docs"] = load_and_chunk_pdfs(uploaded_pdfs)

# Initialize FAISS vector store with embeddings
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None
if uploaded_pdfs and st.session_state["docs"]:
    embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
    index = faiss.IndexFlatL2(len(embeddings.embed_query("test query")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(st.session_state["docs"])
    st.session_state["vector_store"] = vector_store

# Set up retriever and prompt template
if st.session_state["vector_store"]:
    retriever = st.session_state["vector_store"].as_retriever(search_type='mmr', search_kwargs={'k': 3, 'fetch_k': 100, 'lambda_mult': 1})
    prompt = ChatPromptTemplate.from_template("Here is the context from the PDFs: {context}. Now, answer this question: {question}")

# Function to generate chatbot response
def get_response(question):
    # Retrieve relevant documents and format as context
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Prepare prompt using the context and question
    prompt_text = prompt.format(context=context, question=question)
    
    # Generate response with the model
    model_output = model.invoke(prompt_text)
    
    # Parse the response
    response = StrOutputParser().invoke(model_output)
    
    return response

# Chat form for user questions
with st.form("question-form"):
    user_input = st.text_input("Enter your question about the PDFs")  # Moved text_input to this line
    submit_button = st.form_submit_button("Submit")

# Generate and display response
if submit_button and user_input:
    with st.spinner("Generating response..."):
        response = get_response(user_input)
        st.write("### Response:")
        st.write(response)
