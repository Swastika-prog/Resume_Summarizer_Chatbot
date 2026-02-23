# Retrieval Augmented Generation (RAG) System
A Retrieval-Augmented Generation (RAG)–based application that enables context-aware question answering and document summarization using Large Language Models (LLMs).  
The system combines semantic search with generative AI using LangChain, ChromaDB, and Google Gemini to reduce hallucinations and improve factual accuracy.

---

## About the Project
This project demonstrates how Large Language Models can be augmented with external knowledge using vector databases.  
Instead of relying solely on the LLM’s internal knowledge, relevant document chunks are retrieved from a vector store and injected into the prompt, ensuring responses are grounded in actual data.

The system is designed with modular components for ingestion, retrieval, prompt engineering, and generation, making it suitable for real-world AI applications.

---

## Features
- Retrieval-Augmented Generation (RAG) pipeline  
- Semantic document ingestion and chunking  
- Dense embeddings using HuggingFace models  
- Vector storage and similarity search using ChromaDB  
- Context-grounded question answering  
- Document summarization using LLMs  
- Prompt engineering to reduce hallucinations  
- Configurable chunk size and retrieval parameters  
- Modular and extensible codebase  

---

## Tech Stack
- **Language:** Python  
- **LLM Framework:** LangChain  
- **LLM:** Google Gemini  
- **Embeddings:** HuggingFace MiniLM  
- **Vector Database:** ChromaDB  
- **Configuration:** python-dotenv  
- **Frontend (optional):** Streamlit  

---

## Project Structure
```
RAG-System/
│
├── chatbot.py              (RAG-based question answering logic)
├── summarizer.py           (Document summarization pipeline)
├── vector_store.py         (Vector DB creation and retrieval)
├── prompt_template.py      (Prompt templates)
├── chat_prompt.py          (Chat-specific prompts)
├── config.py               (Centralized configuration)
├── requirements.txt        (Project dependencies)
├── .env.example            (Environment variable template)
└── README.md
```


## How to Run the Project

### Step 1: Install dependencies
pip install -r requirements.txt
### Step 2: Configure environment variables
Create a `.env` file in the project root:
GOOGLE_API_KEY=your_google_gemini_api_key
### Step 3: Run the application
python chatbot.py

The system will process the input query, retrieve relevant document chunks from the vector database, and generate a context-aware response using the LLM.

## How It Works
Documents or text data are split into overlapping semantic chunks
Each chunk is converted into dense vector embeddings using HuggingFace models
Embeddings are stored in ChromaDB for efficient similarity search
User queries retrieve the most relevant chunks from the vector database
Retrieved context is injected into a structured prompt
Google Gemini generates responses grounded strictly in retrieved data

## Purpose
This project is intended for learning and demonstrating:
Retrieval-Augmented Generation (RAG) architecture
Vector databases and semantic search
Prompt engineering and hallucination reduction
LLM application development using LangChain
End-to-end AI system design and implementation

## Applications
Document-based question answering
Knowledge assistants and chatbots
Research and study assistants
Website and article summarization
Internal enterprise knowledge systems
