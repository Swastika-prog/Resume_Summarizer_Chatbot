import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 1. Import HuggingFace instead of Google
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain_core.documents import Document
os.environ["USER_AGENT"] = "InsightAI_Research_Bot/1.0"

load_dotenv()
# ADD THIS SECTION:
# Identify yourself to Wikipedia so they don't block you!
os.environ["USER_AGENT"] = "StudentProject/1.0 (test@example.com)"

def ingest_article_to_db(url):
    print(f"Loading article from {url}...")
    
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    
    # 2. Initialize the Local Embedding Model
    # "all-MiniLM-L6-v2" is a famous, fast, and lightweight model.
    print("Initializing embedding model (this might take a minute the first time)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Saving to Chroma Database...")
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("Database successfully created!")
    return vector_store

    # 2. ADD THIS NEW FUNCTION AT THE BOTTOM OF THE FILE
def ingest_text_to_db(text, source_name):
    print(f"Ingesting uploaded text from: {source_name}...")
    
    # Create a "Document" object (this is what LangChain expects)
    doc = Document(page_content=text, metadata={"source": source_name})
    
    # Split into chunks (Reuse the same splitter settings)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents([doc])
    print(f"Split into {len(chunks)} chunks.")
    
    # Initialize the same embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Save to the SAME database folder
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    
    print(f"Successfully saved {source_name} to the database!")
    return True