from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler 
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from local_db import retriever

# 1. Load your .env file
load_dotenv()

# 2. Initialize Langfuse (picks up keys automatically from .env)
langfuse_handler = CallbackHandler()

# 3. Setup your local model
model = OllamaLLM(model="tinyllama")

# 4. Define your prompt template
template = '''
You are an expert in answering questions about a pizza restaurant.

These are some relevant reviews : {reviews}

This is a question for you : {question}
'''
prompt = ChatPromptTemplate.from_template(template)

# 5. Build a unified chain to link retrieval and generation together
rag_chain = (
    {"reviews": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 6. Interactive loop
while True:
    question = input("\nType your question here (q to quit): ")
    if question.lower() == 'q':
        break
        
    if not question.strip():
        continue

    # Run the chain with the callback
    result = rag_chain.invoke(
        question, 
        config={"callbacks": [langfuse_handler]}
    )
    
    print(f"\nAnswer:\n{result}")