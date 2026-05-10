from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from local_db import retriever

model = OllamaLLM(model="tinyllama")

template = '''
You are an expert in answering questions about a pizza restaurant.

These are some relevant reviews : {reviews}

This is a question for you : {question}
'''

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    question = input("Type your question here(q to quit) : ")
    if question == 'q':
        break
    reviews = retriever.invoke(question)
    
    result = chain.invoke({"reviews":reviews, "question":question})
    print(result)