import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 

from backend.prompt_template import research_prompt

load_dotenv()

# ==========================================
# FUNCTION 1: FOR THE DROPDOWN MENU PAPERS
# ==========================================
def generate_summary(paper_name, style, length):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY is missing."

    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.7)
    parser = StrOutputParser()

    try:
        chain = research_prompt | llm | parser
        result = chain.invoke({
            'article_content': paper_name, 
            'style_input': style,
            'length_input': length
        })
        return result
    except Exception as e:
        return f"An error occurred during summarization: {e}"

# ==========================================
# FUNCTION 2: FOR THE WEB SCRAPER URL
# ==========================================
def generate_summary_from_url(url, style, length):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY is missing."

    try:
        # Scrape the website
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        # Split the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        # Grab the first 20 chunks so we get past the menus and into the actual article!
        safe_text_to_summarize = "\n\n".join([chunk.page_content for chunk in chunks[:20]])

        # Setup Model & Parser
        llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.7)
        parser = StrOutputParser()

        # Run the Chain
        chain = research_prompt | llm | parser
        result = chain.invoke({
            'article_content': safe_text_to_summarize,
            'style_input': style,
            'length_input': length
        })
        
        return f"*(Note: Successfully split the article into **{len(chunks)} chunks**! Summarizing the beginning...)*\n\n{result}"
        
    except Exception as e:
        return f"An error occurred while scraping or summarizing: {e}"