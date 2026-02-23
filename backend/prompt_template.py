from langchain_core.prompts import PromptTemplate

# Updated prompt to accept the full text of the scraped article
research_prompt = PromptTemplate(
    template="Please summarize the following article.\n\nArticle Content:\n{article_content}\n\nStyle: {style_input}\nLength: {length_input}\n\nSummary:",
    input_variables=["article_content", "style_input", "length_input"]
)