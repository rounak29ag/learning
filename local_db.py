import os
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings

print("📂 Loading reviews dataset...")
df = pd.read_csv(r"E:\Learning\realistic_restaurant_reviews.csv")
print(f"✅ Loaded {len(df)} reviews from CSV.")

print("🤖 Initializing Ollama embeddings (mxbai-embed-large)...")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_db"
db_not_present = not os.path.exists(db_location)

print(f"🔍 Checking vector database status at '{db_location}'...")
vector_store = Chroma(
    collection_name="restaurants_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if db_not_present:
    print("❌ Vector database not found. Building a new database from scratch...")
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content=row['Title'] + " " + row['Review'],
            metadata={"rating": row['Rating'], "date": row['Date']},
            id=str(i)
        )
        
        ids.append(str(i))
        documents.append(document)
        
    print(f"📄 Structured {len(documents)} text documents.")
    print("🧠 Running local embedding model and saving vectors to disk (this might take a moment)...")
    
    vector_store.add_documents(documents=documents, ids=ids)
    print("💾 Success! All documents have been embedded and stored.")
else:
    print("✅ Existing vector database found! Reusing pre-computed embeddings.")
    
print("🎯 Spinning up retriever interface (configured for Top-5 matches)...")
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
print("🚀 Database setup complete! Retriever is ready for operations.\n" + "="*50)