import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
# 1. Import the necessary libraries for Retrieval
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from backend.chat_prompt import chat_prompt

load_dotenv()

# ==========================================
# SETUP: LOAD THE DATABASE ONCE
# ==========================================
# We load the DB outside the function so we don't reload it 
# every single time you ask a question (makes it faster!)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to the folder you just created
vector_store = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embedding_model
)

# Turn the database into a "Retriever" (a search engine)
# k=3 means "Give me the top 3 most relevant paragraphs"
retriever = vector_store.as_retriever(search_kwargs={"k": 100})


def get_chat_response(paper_name, chat_history, user_input):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY is missing."

    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.7)
    parser = StrOutputParser()

    try:
        # 1. Search the DB for relevant info
        print(f"Searching database for: {user_input}") # Debug print
        docs = retriever.invoke(user_input)
        
        # 2. Combine the search results into one block of text
        # This takes the 3 found paragraphs and joins them with newlines
        retrieved_context = "\n\n".join([doc.page_content for doc in docs])
        
        # 3. Run the Chain with the new context!
        chain = chat_prompt | llm | parser

        print(f"--- RETRIEVED TEXT ({len(docs)} chunks) ---")
        print(retrieved_context[:500]) # Print first 500 chars to check
        print("------------------------------------------------")
        
        result = chain.invoke({
            'context': retrieved_context,   # Inject the database info here
            'chat_history': chat_history,
            'user_input': user_input
        })
        
        return result
        
    except Exception as e:
        return f"An error occurred: {e}"