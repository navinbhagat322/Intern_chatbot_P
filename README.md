# Workflow and Report for PDF-based Chatbot for Conference Call Transcripts

## Project Overview
The goal of this project is to build a chatbot that can respond to user queries based on the content of uploaded PDF documents. The chatbot uses advanced language models, embeddings, and vector storage to retrieve relevant context from the PDFs and provide meaningful responses. This is particularly useful for analyzing and understanding conference call transcripts.

---

## Workflow

### 1. **Project Setup**

#### Steps:
1. Install required dependencies:
   - `streamlit`
   - `langchain`
   - `faiss`
   - `PyMuPDF`
   - `dotenv`
2. Set up the environment:
   - Create a virtual environment and install the necessary packages.
   - Use `.env` to manage environment variables securely.
3. Start the LangChain server for the LLM model (`Ollama`) using the command:
   ```bash
   ollama serve --port 11434
   ```

---

### 2. **File Upload and Parsing**

#### Steps:
1. **Upload PDFs**:
   - Allow users to upload multiple PDF files via the Streamlit interface using the `st.file_uploader()` method.
2. **Parse PDFs**:
   - Use `PyMuPDFLoader` to extract text from PDF files.
   - Use a temporary file system to manage PDF files securely.
   - Delete temporary files after text extraction.

#### Code Snippet:
```python
uploaded_pdfs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
if uploaded_pdfs:
    st.session_state["docs"] = load_and_chunk_pdfs(uploaded_pdfs)
```

---

### 3. **Text Splitting**

#### Steps:
1. Use the `RecursiveCharacterTextSplitter` to divide the extracted text into manageable chunks.
2. Ensure overlapping chunks to maintain context.
3. Define parameters for splitting:
   - `chunk_size=1000`
   - `chunk_overlap=100`

#### Benefits:
- Improves the retriever’s ability to fetch relevant context.
- Ensures that long documents are efficiently indexed.

---

### 4. **Vectorization and Indexing**

#### Steps:
1. **Initialize Embeddings**:
   - Use `OllamaEmbeddings` to generate vector representations of text chunks.
2. **Create FAISS Index**:
   - Initialize a FAISS index with the appropriate dimension based on embedding size.
   - Add document embeddings and map them to their corresponding chunks.
3. **Store Data**:
   - Use `InMemoryDocstore` to keep document metadata for easy retrieval.

#### Code Snippet:
```python
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
```

---

### 5. **Retriever and Prompt Setup**

#### Steps:
1. **Retriever**:
   - Use the FAISS index as the retriever.
   - Configure retrieval parameters:
     - `search_type='mmr'`
     - `search_kwargs={'k': 3, 'fetch_k': 100, 'lambda_mult': 1}`
2. **Prompt Template**:
   - Design a prompt to include retrieved context and user query.
   - Example:
     ```text
     Here is the context from the PDFs: {context}. Now, answer this question: {question}
     ```

#### Benefits:
- Provides accurate and context-aware responses.
- Reduces hallucination by grounding the model with context.

---

### 6. **Chatbot Response Generation**

#### Steps:
1. Retrieve context from the vector store based on user queries.
2. Format the prompt with the retrieved context and the user’s question.
3. Use the language model (`ChatOllama`) to generate a response.
4. Parse and display the response in the Streamlit interface.

#### Code Snippet:
```python
def get_response(question):
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt_text = prompt.format(context=context, question=question)
    model_output = model.invoke(prompt_text)
    response = StrOutputParser().invoke(model_output)
    return response

if submit_button and user_input:
    with st.spinner("Generating response..."):
        response = get_response(user_input)
        st.write("### Response:")
        st.write(response)
```

---

## Project Report

### Challenges
1. **Handling Large PDFs**:
   - Solution: Split text into chunks and use efficient indexing.
2. **Maintaining Context**:
   - Solution: Use overlapping chunks for better contextual understanding.
3. **Embedding Model Performance**:
   - Solution: Optimize embedding dimensions and retrieval parameters.

### Advantages
1. **Dynamic and Scalable**:
   - Supports multiple PDFs.
   - Handles large datasets efficiently.
2. **User-Friendly**:
   - Intuitive Streamlit interface.
3. **Effective Contextual Responses**:
   - Combines embeddings, retrievers, and LLM for accurate answers.

### Future Enhancements
1. Add support for other document formats (e.g., Word, Excel).
2. Incorporate advanced LLMs like GPT-4 for improved responses.
3. Enable persistent storage for FAISS index.
4. Add feedback mechanisms for users to rate responses.

---

## Conclusion
This project demonstrates the integration of modern NLP tools to create an interactive and insightful chatbot. It showcases the power of embeddings, vector search, and LLMs to process complex documents and provide relevant answers. With further enhancements, this tool can become a valuable asset in industries like finance, legal, and research.

