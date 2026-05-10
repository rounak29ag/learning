from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv(r"E:\Learning\realistic_restaurant_reviews.csv")

embeddings = OllamaEmbeddings(model = "mxbai-embed-large")

db_location = "./chroma_db"

db_not_present = not os.path.exists(db_location)

vector_store = Chroma(
    collection_name= "restaurants_reviews",
    persist_directory=db_location,
    embedding_function = embeddings
)

if db_not_present:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content = row['Title'] + " " + row['Review'],
            metadata = {"rating": row['Rating'], "date": row['Date']},
            id = str(i)
        )
        
        ids.append(str(i))
        documents.append(document)
        
    vector_store.add_documents(documents = documents, ids = ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k":5}
)