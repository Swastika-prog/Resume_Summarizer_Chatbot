from backend.vector_store import ingest_article_to_db

# This dictionary maps the dropdown names to real Wikipedia URLs
papers = {
    "Attention Is All You Need": "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
    "BERT": "https://en.wikipedia.org/wiki/BERT_(language_model)",
    "GPT-3": "https://en.wikipedia.org/wiki/GPT-3",
    "Diffusion Models": "https://en.wikipedia.org/wiki/Diffusion_model"
}

print("🚀 Starting Knowledge Base Builder...")

for name, url in papers.items():
    print(f"\nProcessing: {name}...")
    # We call the function you already wrote!
    ingest_article_to_db(url)
    print(f"✅ Learned about {name}!")

print("\n🎉 All papers ingested! Your chatbot is now an expert on all 4 topics.")