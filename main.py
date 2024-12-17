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

# Check if the API keys were provided and saved
if "api_keys_set" not in st.session_state:
    st.session_state.api_keys_set = False
    st.session_state.llm = None
    st.session_state.file_uploaded = False
    st.session_state.model_chosen = False

# Text inputs for Google and Groq API keys
google_api_key = st.text_input("Enter your Google Cloud API key:", type="password")
groq_api_key = st.text_input("Enter your Groq API key:", type="password")

# Reset the process if the API keys change
if google_api_key and groq_api_key and not st.session_state.api_keys_set:
    if st.button("Submit API"):
        env_data = f"GROQ_API_KEY={groq_api_key}\nGOOGLE_API_KEY={google_api_key}"
        with open(".env", "w") as file:
            file.write(env_data)
        load_dotenv()
        st.session_state.api_keys_set = True
        st.session_state.llm = None  # Reset any previous model
        st.session_state.file_uploaded = False
        st.session_state.model_chosen = False
        st.success("API keys have been saved successfully!")

# Ensure the user is prompted for API keys before continuing
if not st.session_state.api_keys_set:
    st.warning("Please enter both Google and Groq API keys to proceed.")
else:

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
    
            # Clean up the temporary file
            os.remove(temp_pdf_path)
        except Exception as e:
            st.error(f"Error during vector database creation: {e}")
    # PDF file uploader (only show after API keys are set)
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file:
        if st.button("Embed File"):
            if "vectors" not in st.session_state:
                vector_db(uploaded_file)
            st.session_state.file_uploaded = True
            st.success("PDF embedded successfully!")

    # Only show model choice after file has been uploaded
    if st.session_state.file_uploaded:
        model_choice = st.selectbox(
            "Select the LLM model:",
            options=["mixtral-8x7b-32768", "gemma2-9b-it", "gemma-7b-it", "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
                     "llama-guard-3-8b", "llama3-70b-8192", "llama3-8b-8192"],
            help="Choose the model to use for your chatbot."
        )

        # Initialize LLM with Groq API key and selected model
        if model_choice and not st.session_state.model_chosen:
            if st.button("Choose Model"):
                st.session_state.llm = ChatGroq(model=model_choice)
                st.session_state.model_chosen = True
                st.success(f"Model {model_choice} loaded successfully!")

    # Allow the user to send a message only if the model is chosen
    if st.session_state.model_chosen:
        user_query = st.chat_input("Enter your question:")

        if user_query:
            if st.button("Send"):
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
