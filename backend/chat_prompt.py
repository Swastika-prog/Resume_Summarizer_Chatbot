from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages([
    # Updated System Message: Now accepts 'context' from the database
    ("system", "You are an AI research assistant. "
               "Use the following pieces of retrieved context to answer the user's question. "
               "If you don't know the answer, just say that you don't know.\n\n"
               "Context:\n{context}"),
    
    MessagesPlaceholder(variable_name="chat_history"),
    
    ("human", "{user_input}")
])