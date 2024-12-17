import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import tempfile

# Streamlit UI
st.title("Groq-PDF-Chatbot")

# Text inputs for Google and Groq API keys
google_api_key = st.text_input("Enter your Google Cloud API key:", type="password")
groq_api_key = st.text_input("Enter your Groq API key:", type="password")

# Save API keys to .env if provided
if google_api_key and groq_api_key:
    st.session_state.google_api_key = google_api_key
    st.session_state.groq_api_key = groq_api_key

    env_data = f"GROQ_API_KEY={groq_api_key}\nGOOGLE_API_KEY={google_api_key}"
    with open(".env", "w") as file:
        file.write(env_data)
    # Load environment variables
    load_dotenv()
else:
    st.warning("Please enter both Google and Groq API keys.")

# Model selection dropdown
model_choice = st.selectbox(
    "Select the LLM model:",
    options=["llama-3.2-3b-preview", "llama-7b", "llama-13b"],
    help="Choose the model to use for your chatbot.",
)

# Initialize LLM with Groq API key and selected model
if "llm" not in st.session_state:
    if groq_api_key:
        st.session_state.llm = ChatGroq(model=model_choice)
        st.success(f"Model {model_choice} loaded successfully!")
    else:
        st.error("Groq API key is missing!")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Questions: {input}
    """
)

# Function to embed PDF data into the vector store
def vector_db(pdf_file):
    try:
        # Save the uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_file.read())
            temp_pdf_path = temp_pdf.name
        
        # Load and split the PDF file using the temporary path
        st.session_state.embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        pdf_loader = PyPDFLoader(temp_pdf_path)
        doc_text = pdf_loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_text = st.session_state.text_splitter.split_documents(doc_text)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_text, st.session_state.embedding)
        st.success("PDF data embedded successfully!")

        # Clean up the temporary file
        os.remove(temp_pdf_path)
    except Exception as e:
        st.error(f"Error during vector database creation: {e}")

# PDF file uploader
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# Handle PDF upload and embedding
if uploaded_file:
    if "vectors" not in st.session_state:
        vector_db(uploaded_file)

    # User input for queries
    user_query = st.chat_input("Enter your question:")

    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        try:
            # Create document retrieval chain
            document_chain = create_stuff_documents_chain(st.session_state.llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            chain_llm = create_retrieval_chain(retriever, document_chain)

            # Generate response
            response = chain_llm.invoke({"input": user_query})

            with st.chat_message("assistant"):
                st.write(response["answer"])
        except Exception as e:
            st.error(f"An error occurred while processing your query: {e}")

