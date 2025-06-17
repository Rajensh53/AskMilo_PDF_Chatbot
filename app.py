import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is not set. Please check your .env file and path.")

st.title("AskMilo 💡💬")
st.write("--A curious AI assistant that provied internal insights from PDF--")
st.write("You can interact with the chatbot by entering your queries below.")

# Session state to store chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_vectorstore():
    pdf_name = "./AINotes.pdf"
    loaders = [PyPDFLoader(pdf_name)]
    # Create chunks, aka vector database–Chromadb
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)
    return index.vectorstore

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["content"])
    else:
        st.chat_message("assistant").markdown(message["content"])


prompt = st.chat_input("Enter your query: ")

if prompt:
    st.chat_message("user").markdown(prompt)
    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    groq_sys_prompt = ChatPromptTemplate.from_template("""You are an AI assistant designed to read, analyze, and extract information from given PDF document. summarize the key contents, and answer any questions the user asks based on the document.
                                                         Always refer only to the information within the PDF and respond clearly and concisely. Answer the following Question: {user_prompt}.
                                                         Start the answer directly. No small talk please""")
    
    # Create a Groq chat model instance
    model = "llama3-8b-8192"
    groq_chat = ChatGroq(
        model_name = model
    )
    
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Vectorstore is not available. Please check the document loader and embeddings.")
    
        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )


        result = chain({"query": prompt})
        response = result["result"]


        # Model Response
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    except Exception as e:
        st.error(f"An error occurred: {e}")







