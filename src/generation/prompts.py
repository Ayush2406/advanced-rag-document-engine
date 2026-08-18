from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are an assistant for question-answering tasks.You will be given context related to the human query
your task is to answer to the query with the help of only the context no answers should be beyond the context.If you cannot
answer from the provided context than state clearly: 'I cannot answer this question based on the provided Documents"""

HUMAN_PROMPT = """Context:
{context}

Question:
{question}

Helpful Answer:"""

def get_rag_prompt()->ChatPromptTemplate:
    
    return ChatPromptTemplate.from_messages([
        ("system",SYSTEM_PROMPT),
        ("human",HUMAN_PROMPT)
    ])
