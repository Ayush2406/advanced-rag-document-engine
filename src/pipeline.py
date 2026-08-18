from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from typing import List

from src.retrieval.vector_store import VectorStore
from src.generation.prompts import get_rag_prompt
from src.generation.llm_service import get_llm
from src.config import settings

def format_docs(docs: List[Document])->str:
    """Combines retrieved document objects into single context string"""
    return "\n\n".join(doc.page_content for doc in docs)

def rag_pipeline(vector_store:VectorStore,top_k:int = settings.TOP_K):
    """
    Constructs and returns the LCEL RAG chain
    Question -> Retriever -> Prompt -> LLM -> Final String Answer 
    """
    
    retriever = vector_store.as_retriever(search_kwargs={"k":top_k})
    prompt = get_rag_prompt()
    llm = get_llm(temperature=0.0)
    
    # LCEL Chain Assembly
    
    rag_chain = (
        {
            "context":retriever | format_docs,
            "question":RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser() # same as result.content
    )
    return rag_chain