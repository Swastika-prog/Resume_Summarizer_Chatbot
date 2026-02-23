import streamlit as st
import sys
import os
import shutil
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import fitz  # PyMuPDF

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Backend Imports
from backend.summarizer import generate_summary, generate_summary_from_url
from backend.chatbot import get_chat_response
from backend.vector_store import ingest_text_to_db

# 1. Page Config
st.set_page_config(page_title="InsightAI | Research Partner", page_icon="🧠", layout="wide")

# 2. Advanced Styling (Glassmorphism + Smooth Transitions)
st.markdown("""
    <style>
    [data-testid="stChatMessage"] { border-radius: 15px; border: 1px solid rgba(0,0,0,0.05); padding: 20px; }
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #4F8BF9 , #8E9EAB); }
    .metric-container { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; }
    </style>
""", unsafe_allow_html=True)

# 3. Helpers
def get_pdf_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])

def get_db_info():
    """Extracts stats and source names from the database"""
    try:
        if os.path.exists("./chroma_db"):
            embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = Chroma(persist_directory="./chroma_db", embedding_function=embed)
            collection = db.get()
            # Get unique source names from metadata
            sources = list(set([m.get('source', 'Unknown') for m in collection['metadatas']]))
            return db._collection.count(), sources
        return 0, []
    except: return 0, []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 4. SIDEBAR: THE INFORMATION HUB
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712009.png", width=60)
    st.title("InsightAI Dashboard")
    st.caption("Real-time Intelligence Monitor")
    st.divider()
    
    # --- Live Metrics ---
    chunks, source_list = get_db_info()
    
    st.markdown("### 📊 Knowledge Metrics")
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Data Chunks", chunks)
    m_col2.metric("Documents", len(source_list))
    
    # Brain IQ Logic
    iq_val = "Novice" if chunks < 100 else "Capable" if chunks < 400 else "Expert"
    st.write(f"**Current Intelligence Level:** {iq_val}")
    st.progress(min(chunks / 500, 1.0))
    
    st.divider()
    
    # --- Document Library ---
    st.markdown("### 📚 Loaded Library")
    if source_list:
        for src in source_list:
            st.caption(f"📄 {src}")
    else:
        st.write("No external documents loaded yet.")

    st.divider()
    
    # --- Controls ---
    with st.expander("🛠️ Advanced Controls"):
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        if st.button("☢️ Reset Global Brain", type="secondary", use_container_width=True):
            if os.path.exists("./chroma_db"): shutil.rmtree("./chroma_db")
            st.session_state.chat_history = []
            st.rerun()

# 5. MAIN CONTENT
st.title("🧠 InsightAI Assistant")
tab1, tab2 = st.tabs(["📄 **Summarizer & Ingestion**", "💬 **Active Chatbot**"])

# --- TAB 1: SUMMARIZER & DATA UPLOAD ---
with tab1:
    st.markdown("#### Add Knowledge to the System")
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("**1. Select Source**")
        input_type = st.radio("Choose how to input data:", ["URL", "Pre-loaded", "Upload PDF"], horizontal=True, label_visibility="collapsed")
        
        user_input, uploaded_file = None, None
        if input_type == "URL": 
            user_input = st.text_input("🔗 Article URL:", placeholder="https://example.com/article")
        elif input_type == "Pre-loaded": 
            user_input = st.selectbox("📚 Choose Paper:", ["Attention Is All You Need", "BERT", "GPT-3", "Diffusion Models"])
        else: 
            uploaded_file = st.file_uploader("📂 Choose a PDF file", type=['pdf'])

    with col2:
        st.markdown("**2. Configure Output**")
        style_input = st.select_slider("Explanation Depth", options=["Beginner", "Technical", "Code", "Math"])
        length_input = st.select_slider("Output Length", options=["Short", "Medium", "Long"])
        
        st.write("")
        if st.button('✨ Process, Summarize & Memorize', use_container_width=True):
            if (input_type == "URL" and not user_input) or (input_type == "Upload PDF" and not uploaded_file):
                st.error("Please provide a valid source.")
            else:
                with st.spinner("🧠 AI is reading and indexing..."):
                    try:
                        summary = ""
                        if input_type == "URL": 
                            summary = generate_summary_from_url(user_input, style_input, length_input)
                        elif input_type == "Pre-loaded": 
                            summary = generate_summary(user_input, style_input, length_input)
                        elif uploaded_file:
                            text = get_pdf_text(uploaded_file)
                            summary = generate_summary(text, style_input, length_input)
                            ingest_text_to_db(text, uploaded_file.name)
                            st.toast(f"Brain Updated: {uploaded_file.name}")
                        
                        if summary:
                            st.success("Analysis Complete!")
                            with st.container():
                                st.markdown("### 📝 Research Summary")
                                st.info(summary)
                    except Exception as e:
                        st.error(f"Processing Error: {e}")

# --- TAB 2: CHATBOT ---
with tab2:
    st.markdown("#### Chat with the Collective Knowledge Base")
    # Show what papers the bot knows about
    if source_list:
        st.caption(f"Bot currently has context from: {', '.join(source_list[:3])}...")
    
    # Conversation Display
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        avatar = "👤" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar): 
            st.write(msg.content)

    # Input handling
    if prompt := st.chat_input("Ask a specific question..."):
        with st.chat_message("user", avatar="👤"): 
            st.write(prompt)
            
        with st.spinner("Searching memory..."):
            response = get_chat_response("All", st.session_state.chat_history, prompt)
            
        with st.chat_message("assistant", avatar="🤖"): 
            st.write(response)
            
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.chat_history.append(AIMessage(content=response))